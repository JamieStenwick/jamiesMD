// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_TRSM_EXECUTE_HPP
#define CUBLASDX_DATABASE_TRSM_EXECUTE_HPP

// trsm.hpp provides the forward declarations of thread_impl/warp_impl/cta_impl
// and the accessor factory helpers (make_ro_accessor etc.).
#include "cublasdx/database/trsm/trsm.hpp"
// tensor.hpp is not included in the fatbin compilation path, so block_execute
// and thread_execute (which use cublasdx::tensor<>) live here, not in trsm.hpp.
#include "cublasdx/detail/tensor.hpp"

namespace cublasdx::detail::trsm {
    template<int AlignmentInBytes, class BEngine, class BLayout>
    __device__ __forceinline__ auto copy_to_rmem(const cublasdx::tensor<BEngine, BLayout>& tensor_b) {
        auto frg = cute::make_fragment_like(tensor_b);
        
        auto tmp_src = cute::make_tensor(tensor_b.data(), cute::coalesce(tensor_b.layout()));
        auto tmp_dst = cute::make_tensor(frg.data(), cute::coalesce(frg.layout()));
        cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentInBytes * 8> {}, tmp_src, tmp_dst);
        return frg;
    }

    template<bool Pred, class Tensor>
    CUBLASDX_HOST_DEVICE constexpr auto transpose_if(const Tensor & tensor) {
        return cute::conditional_return<Pred>(transpose_view(tensor), tensor);
    }

    template<unsigned   M,
             unsigned   N,
             side       Side,
             diag       Diag,
             fill_mode  Fill,
             unsigned   NT,
             unsigned   BPB,
             int        BAlignmentInBytes,
             class AEngine, class ALayout,
             class BEngine, class BLayout>
    inline __device__ void block_execute(const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                                         cublasdx::tensor<BEngine, BLayout>&       tensor_b,
                                         unsigned                                   thread_id) {
        using T = typename AEngine::value_type;

        constexpr bool is_right    = (Side == side::right);
        constexpr bool unit_diag  = (Diag == diag::unit);
        constexpr bool even_warps = (NT % 32 == 0);
        constexpr int  num_warps  = NT / 32;
        constexpr unsigned impl_M = is_right ? N : M;  // dimension of the triangular system
        constexpr unsigned impl_N = is_right ? M : N;  // number of RHS columns

        // forward ↔ lower-triangular forward substitution
        constexpr bool fwd = is_right != (Fill == fill_mode::lower);

        auto normalized_tensor_a = transpose_if<is_right>(tensor_a);
        auto normalized_tensor_b = transpose_if<is_right>(tensor_b);

        if constexpr (BPB > 1) {
            // ------------------------------------------------------------------
            // Batched: tensor_a / tensor_b carry a batch mode as their 3rd index.
            // Slice each batch with CuTe: tensor(cute::_, cute::_, batch).
            // ------------------------------------------------------------------
            if (BPB * impl_N >= NT || impl_M <= 8) {
                // At least 1 RHS per thread, or tiny matrix: each thread owns one column.
                // Left: column j of B → tensor_b(cute::_, col, batch)
                // Right: row col of B (algo "column col") → tensor_b(col, cute::_, batch)
                for (int t = (int)thread_id; t < (int)(BPB * impl_N); t += (int)NT) {
                    int batch    = t / (int)impl_N;
                    int col      = t % (int)impl_N;
                    auto ab      = normalized_tensor_a(cute::_, cute::_, batch);
                    auto b_slice = normalized_tensor_b(cute::_, col, batch);
                    constexpr tensor_readonly_accessor<T> a_acc = trsm_immutable_tensor_accessor<cute::remove_cvref_t<decltype(ab)>>;   
                    constexpr tensor_readwrite_accessor<T> b_acc = trsm_mutable_vector_accessor<cute::remove_cvref_t<decltype(b_slice)>>;
                    auto rmem = copy_to_rmem<BAlignmentInBytes>(b_slice);
                    trsm::thread_impl<T>(reinterpret_cast<void const*>(&ab), reinterpret_cast<void *>(&b_slice), cute::raw_pointer_cast(rmem.data()), impl_M, 1, unit_diag, fwd, a_acc, b_acc);
                }
            } else if constexpr (even_warps && BPB * impl_N >= (unsigned)num_warps) {
                // At least 1 RHS per warp - same single-column slice pattern.
                unsigned warp_id = thread_id / 32;
                unsigned lane_id = thread_id % 32;
                for (int t = (int)warp_id; t < (int)(BPB * impl_N); t += num_warps) {
                    int batch    = t / (int)impl_N;
                    int col      = t % (int)impl_N;
                    auto ab      = normalized_tensor_a(cute::_, cute::_, batch);
                    auto b_slice = normalized_tensor_b(cute::_, col, batch);
                    constexpr tensor_readonly_accessor<T> a_acc = trsm_immutable_tensor_accessor<cute::remove_cvref_t<decltype(ab)>>;
                    constexpr tensor_readwrite_accessor<T> b_acc = trsm_mutable_vector_accessor<cute::remove_cvref_t<decltype(b_slice)>>;
                    constexpr unsigned nrows = (impl_M + 31) / 32;
                    T rmem1[nrows]; T rmem2[1]; T rmem3[impl_M];
                    trsm::warp_impl<T>(reinterpret_cast<void const*>(&ab), reinterpret_cast<void *>(&b_slice), lane_id, rmem1, rmem2, rmem3,
                                       32, impl_M, 1, unit_diag, fwd, a_acc, b_acc);
                }
            } else {
                // Full CTA per batch element.
                for (int batch = 0; batch < (int)BPB; ++batch) {
                    auto ab    = normalized_tensor_a(cute::_, cute::_, batch);
                    auto bb    = normalized_tensor_b(cute::_, cute::_, batch);
                    constexpr tensor_readonly_accessor<T> a_acc = trsm_immutable_tensor_accessor<cute::remove_cvref_t<decltype(ab)>>;
                    constexpr tensor_readwrite_accessor<T> b_acc = trsm_mutable_tensor_accessor<cute::remove_cvref_t<decltype(bb)>>;
                    constexpr unsigned nrows = (impl_M + NT - 1) / NT;
                    T rmem1[nrows * impl_N]; T rmem2[impl_N];
                    trsm::cta_impl<T>(reinterpret_cast<void const*>(&ab), reinterpret_cast<void *>(&bb), thread_id, rmem1, rmem2,
                                      impl_M, impl_N, NT, unit_diag, fwd, a_acc, b_acc);
                }
            }

        } else {
            // BPB == 1: tensor_a is 2D - safe to create a 2D accessor here.
            constexpr tensor_readonly_accessor<T> acc_a = trsm_immutable_tensor_accessor<cute::remove_cvref_t<decltype(normalized_tensor_a)>>;

            if constexpr (NT / impl_N > 40 && impl_M > 32) {
                // ------------------------------------------------------------------
                // Many threads per RHS and tall matrix: full-CTA synchronous solve.
                // ------------------------------------------------------------------
                constexpr unsigned nrows = (impl_M + NT - 1) / NT;
                constexpr tensor_readwrite_accessor<T> acc_b = trsm_mutable_tensor_accessor<cute::remove_cvref_t<decltype(normalized_tensor_b)>>;
                T rmem1[nrows * impl_N]; T rmem2[impl_N];
                trsm::cta_impl<T>(reinterpret_cast<void const*>(&normalized_tensor_a), reinterpret_cast<void *>(&normalized_tensor_b), thread_id, rmem1, rmem2,
                                impl_M, impl_N, NT, unit_diag, fwd, acc_a, acc_b);

            } else if constexpr (NT >= 32 && impl_N < NT && impl_M >= 8) {
                // ------------------------------------------------------------------
                // Warp-level dispatch: split RHS columns across warps / sub-warps.
                // Left:  local_tile shape (impl_M, sub_warp_N), coord (0, tile_idx)
                // Right: local_tile shape (sub_warp_N, impl_M), coord (tile_idx, 0)
                // ------------------------------------------------------------------
                unsigned warp_id = thread_id / 32;
                unsigned lane_id = thread_id % 32;

                if constexpr (even_warps && impl_N % (unsigned)num_warps == 0
                            && impl_N >= (unsigned)num_warps) {
                    constexpr unsigned N_per_warp   = impl_N / (unsigned)num_warps;
                    constexpr unsigned num_sub_warp = pow2_divisor(N_per_warp);
                    static_assert(num_sub_warp == 1  || num_sub_warp == 2  || num_sub_warp == 4 ||
                                num_sub_warp == 8  || num_sub_warp == 16 || num_sub_warp == 32);
                    static_assert(N_per_warp % num_sub_warp == 0);
                    constexpr unsigned sub_warp_NT = 32 / num_sub_warp;
                    constexpr unsigned sub_warp_N  = N_per_warp / num_sub_warp;
                    unsigned tile_idx = thread_id / sub_warp_NT;

                    constexpr unsigned nrows = (impl_M + sub_warp_NT - 1) / sub_warp_NT;
                    T rmem1[nrows * sub_warp_N]; T rmem2[sub_warp_N]; T rmem3[impl_M];

                    auto b_tile = cute::local_tile(normalized_tensor_b,
                                                   cute::make_shape(cute::Int<impl_M>{}, cute::Int<sub_warp_N>{}),
                                                   cute::make_coord(0, tile_idx));

                    constexpr tensor_readwrite_accessor<T> acc_b = trsm_mutable_tensor_accessor<cute::remove_cvref_t<decltype(b_tile)>>;
                    trsm::warp_impl<T>(reinterpret_cast<void const*>(&normalized_tensor_a), reinterpret_cast<void *>(&b_tile), lane_id, rmem1, rmem2, rmem3,
                                    sub_warp_NT, impl_M, sub_warp_N, unit_diag, fwd, acc_a, acc_b);
                } else {
                    // Only use warp 0
                    if (warp_id == 0) {
                        constexpr unsigned pow2_div_N  = pow2_divisor(impl_N);
                        constexpr unsigned num_sub_warp = (pow2_div_N > 32u) ? 32u : pow2_div_N;
                        constexpr unsigned sub_warp_NT  = 32 / num_sub_warp;
                        constexpr unsigned sub_warp_N   = impl_N / num_sub_warp;
                        unsigned tile_idx = lane_id / sub_warp_NT;

                        constexpr unsigned nrows = (impl_M + sub_warp_NT - 1) / sub_warp_NT;
                        T rmem1[nrows * sub_warp_N]; T rmem2[sub_warp_N]; T rmem3[impl_M];

                        auto b_tile = cute::local_tile(normalized_tensor_b,
                                                       cute::make_shape(cute::Int<impl_M>{}, cute::Int<sub_warp_N>{}),
                                                       cute::make_coord(0, tile_idx));
                        constexpr tensor_readwrite_accessor<T> acc_b = trsm_mutable_tensor_accessor<cute::remove_cvref_t<decltype(b_tile)>>;
                        trsm::warp_impl<T>(reinterpret_cast<void const*>(&normalized_tensor_a), reinterpret_cast<void *>(&b_tile), lane_id, rmem1, rmem2, rmem3,
                                        sub_warp_NT, impl_M, sub_warp_N, unit_diag, fwd, acc_a, acc_b);
                    }
                }

            } else if constexpr (2 * impl_N >= NT || impl_M < 8) {
                // ------------------------------------------------------------------
                // Large N or tiny matrix: each thread owns its own column(s).
                // ------------------------------------------------------------------
                if constexpr (impl_N % NT == 0) {
                    // Evenly divisible: give each thread exactly thread_N columns as a tile.
                    constexpr unsigned thread_N = impl_N / NT;
                    auto b_tile = cute::local_tile(normalized_tensor_b,
                                                   cute::make_shape(cute::Int<impl_M>{}, cute::Int<thread_N>{}),
                                                   cute::make_coord(0, thread_id));
                    constexpr tensor_readwrite_accessor<T> acc_b = trsm_mutable_tensor_accessor<cute::remove_cvref_t<decltype(b_tile)>>;
                    auto rmem = copy_to_rmem<BAlignmentInBytes>(b_tile);
                    trsm::thread_impl<T>(reinterpret_cast<void const*>(&normalized_tensor_a), reinterpret_cast<void *>(&b_tile), cute::raw_pointer_cast(rmem.data()), impl_M, thread_N, unit_diag, fwd, acc_a, acc_b);
                } else {
                    // Round-robin: each thread iterates over its columns one at a time.
                    for (int j = (int)thread_id; j < (int)impl_N; j += (int)NT) {
                        auto b_slice = normalized_tensor_b(cute::_, j);
                        constexpr tensor_readwrite_accessor<T> acc_b = trsm_mutable_vector_accessor<cute::remove_cvref_t<decltype(b_slice)>>;
                        auto rmem = copy_to_rmem<BAlignmentInBytes>(b_slice);
                        trsm::thread_impl<T>(reinterpret_cast<void const*>(&normalized_tensor_a), reinterpret_cast<void *>(&b_slice), cute::raw_pointer_cast(rmem.data()), impl_M, 1, unit_diag, fwd, acc_a, acc_b);
                    }
                }
            } else {
                // Fallback: full-CTA synchronous solve.
                constexpr unsigned nrows = (impl_M + NT - 1) / NT;
                constexpr tensor_readwrite_accessor<T> acc_b = trsm_mutable_tensor_accessor<cute::remove_cvref_t<decltype(normalized_tensor_b)>>;
                T rmem1[nrows * impl_N]; T rmem2[impl_N];
                trsm::cta_impl<T>(reinterpret_cast<void const*>(&normalized_tensor_a), reinterpret_cast<void *>(&normalized_tensor_b), thread_id, rmem1, rmem2,
                                impl_M, impl_N, NT, unit_diag, fwd, acc_a, acc_b);
            }
        } // end BPB == 1
    }

    // ------------------------------------------------------------------
    // thread_execute: single-thread triangular solve (no synchronisation).
    // ------------------------------------------------------------------
    template<unsigned   M,
             unsigned   N,
             side       Side,
             diag       Diag,
             fill_mode  Fill,
             int        BAlignmentInBytes,
             class AEngine, class ALayout,
             class BEngine, class BLayout>
    inline __device__ void thread_execute(const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                                          cublasdx::tensor<BEngine, BLayout>&       tensor_b) {
        using T = typename AEngine::value_type;

        constexpr bool is_right   = (Side == side::right);
        constexpr bool unit_diag = (Diag == diag::unit);
        constexpr unsigned impl_M = is_right ? N : M;
        constexpr unsigned impl_N = is_right ? M : N;
        constexpr bool fwd = is_right != (Fill == fill_mode::lower);

        auto normalized_tensor_a = transpose_if<is_right>(tensor_a);
        auto normalized_tensor_b = transpose_if<is_right>(tensor_b);

        constexpr tensor_readonly_accessor<T> acc_a = trsm_immutable_tensor_accessor<cute::remove_cvref_t<decltype(normalized_tensor_a)>>;
        constexpr tensor_readwrite_accessor<T> acc_b = trsm_mutable_tensor_accessor<cute::remove_cvref_t<decltype(normalized_tensor_b)>>;

        auto rmem = copy_to_rmem<BAlignmentInBytes>(normalized_tensor_b);
        trsm::thread_impl<T>(reinterpret_cast<void const*>(&normalized_tensor_a), reinterpret_cast<void *>(&normalized_tensor_b), cute::raw_pointer_cast(rmem.data()), impl_M, impl_N, unit_diag, fwd, acc_a, acc_b);
    }

} // namespace cublasdx::detail::trsm

#endif // CUBLASDX_DATABASE_TRSM_EXECUTE_HPP
