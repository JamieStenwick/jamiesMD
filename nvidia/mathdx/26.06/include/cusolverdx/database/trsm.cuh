// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_TRSM_CUH
#define CUSOLVERDX_DATABASE_TRSM_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/indexing.cuh"
#include "cusolverdx/database/trsm_db.cuh"

namespace cusolverdx::detail::trsm {
    
    template<class T>
    __device__ void thread_impl(const T* A, const unsigned lda, T* B, const unsigned ldb, T* rmem, const unsigned M, const unsigned N, const bool unit_diag, const bool conj_a, const bool trans_b, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool forward);
    // Designed for small N
    template<class T>
    __device__ void warp_impl(const T* A, const unsigned lda, T* B, const unsigned ldb, unsigned thread_id, T* rmem1, T* rmem2, T* rmem3, const unsigned NT, const unsigned M, const unsigned N, const bool unit_diag, const bool conj_a, const bool trans_b, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool forward);    
    // Designed for small N
    template<class T>
    __device__ void cta_impl(const T* A, const unsigned lda, T* B, const unsigned ldb, unsigned thread_id, T* rmem1, T* rmem2, const unsigned M, const unsigned N, const unsigned NT, const bool unit_diag, const bool conj_a, const bool trans_b, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool forward);


    // B is MxN
    // if side left, A is MxM, else A is NxN
    // Transpose is evaluated after Fill
    template<class T, unsigned M, unsigned N, side Side, diag Diag, transpose Transpose, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned BPB>
    inline __device__ void block_execute(const T* A, const unsigned lda, const unsigned strideA, T* B, const unsigned ldb, const unsigned strideB, const unsigned thread_id) {
        constexpr bool is_left   = (Side == side::left);
        constexpr bool unit_diag = (Diag == diag::unit);

        constexpr bool even_warps = NT % 32 == 0;
        constexpr int  num_warps  = NT / 32;

        // If right side requested, transpose both matrices (but never conjugate)
        constexpr bool     conj_a  = Transpose == transpose::conj_transposed;
        constexpr bool     trans_b = !is_left;
        constexpr unsigned impl_M  = is_left ? M : N;
        constexpr unsigned impl_N  = is_left ? N : M;
        // NB., we don't need to pass whether A is transposed to the implementation
        // fwd indicates which direction to process the diagonal.  Fill/Arrange indicate how to access the off-diagonal elements

        // left, non-transposed, lower uses forward substitution
        // toggling any property toggles the type of substitution
        // Thus, we use XOR to check if we need backward substitution
        // != is equivalent to XOR for bools
        constexpr bool fwd = !is_left != (Transpose != transpose::non_transposed) != (Fill == fill_mode::lower);

        if constexpr (BPB > 1u) {
            // Handle batched dispatch separately from non-batched

            // TODO try to do multiple RHSs together where possible
            // matrix size 8 x 8 is faster handled by the thread_impl
            if (BPB * impl_N >= NT || impl_M <= 8u) {
                // At least 1 RHS per thread or tiny matrices
                for (int j = thread_id; j < int(BPB * impl_N); j += NT) {
                    unsigned batch = j / impl_N;
                    unsigned RHS   = j % impl_N;

                    auto Aj = A + batch * strideA;
                    auto Bj = B + batch * strideB;
                    auto bj = &index<T>(Bj, ldb, 0, RHS, trans_b, ArrangeB);

                    T rmem[impl_M];

                    trsm::thread_impl<T>(Aj, lda, bj, ldb, rmem, impl_M, 1, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                }
            } else if constexpr (even_warps && BPB * impl_N >= num_warps) {
                // At least 1 RHS per warp
                unsigned warp_id = thread_id / 32;
                unsigned lane_id = thread_id % 32;
                for (int j = warp_id; j < int(BPB * impl_N); j += num_warps) {
                    unsigned batch = j / impl_N;
                    unsigned RHS   = j % impl_N;

                    auto Aj = A + batch * strideA;
                    auto Bj = B + batch * strideB;
                    auto bj = &index<T>(Bj, ldb, 0, RHS, trans_b, ArrangeB);

                    constexpr unsigned nrows = (impl_M + 31) / 32;

                    T rmem1[nrows];
                    T rmem2[1];
                    T rmem3[impl_M];
                    trsm::warp_impl<T>(Aj, lda, bj, ldb, lane_id, rmem1, rmem2, rmem3, 32, impl_M, 1, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                }
            } else {
                // Just use the full CTA
                for (int j = 0; j < int(BPB); ++j) {
                    auto Aj = A + j * strideA;
                    auto Bj = B + j * strideB;

                    constexpr unsigned nrows = (impl_M + NT - 1) / NT;

                    T rmem1[nrows * impl_N];
                    T rmem2[impl_N];
                    trsm::cta_impl<T>(Aj, lda, Bj, ldb, thread_id, rmem1, rmem2, impl_M, impl_N, NT, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                }
            }

        } else if constexpr (NT / impl_N > 40 && impl_M > 32) {
            // Have many threads per RHS and more rows than a single warp's threads.
            // Use full CTA synchronously
            constexpr unsigned nrows = (impl_M + NT - 1) / NT;

            T rmem1[nrows * impl_N];
            T rmem2[impl_N];
            trsm::cta_impl<T>(A, lda, B, ldb, thread_id, rmem1, rmem2, impl_M, impl_N, NT, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);

        } else if constexpr (NT >= 32u && impl_N < NT && impl_M >= 8u) {
            // Have at least a warp, fewer RHSs than threads, and the matrix isn't tiny
            // Split the problem across warps
            unsigned warp_id = thread_id / 32;
            unsigned lane_id = thread_id % 32;
            if constexpr (even_warps && impl_N % num_warps == 0 && impl_N >= num_warps) {
                // Can split the RHS's nicely

                // NB, because impl_N < NT, we have N_per_war < 32
                constexpr unsigned N_per_warp = impl_N / num_warps;
                // Compute the largest power of two for which N_per_warp is divisible, then use that many subwarps
                constexpr unsigned num_sub_warp = pow2_divisor(N_per_warp);
                static_assert(num_sub_warp == 1 || num_sub_warp == 2 || num_sub_warp == 4 || num_sub_warp == 8 || num_sub_warp == 16 || num_sub_warp == 32);
                static_assert(N_per_warp % num_sub_warp == 0);
                constexpr unsigned sub_warp_NT = 32 / num_sub_warp;
                constexpr unsigned sub_warp_N  = N_per_warp / num_sub_warp;

                unsigned j  = sub_warp_N * (thread_id / sub_warp_NT);
                T*       bj = &index<T>(B, ldb, 0, j, trans_b, ArrangeB);

                constexpr unsigned nrows = (impl_M + sub_warp_NT - 1) / sub_warp_NT;

                T rmem1[nrows * sub_warp_N];
                T rmem2[sub_warp_N];
                T rmem3[impl_M];
                trsm::warp_impl<T>(A, lda, bj, ldb, lane_id, rmem1, rmem2, rmem3, sub_warp_NT, impl_M, sub_warp_N, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
            } else {
                // Just use a single warp
                if (warp_id == 0) {
                    // Compute the largest power of two for which N_per_warp is divisible, then use that many subwarps (has to be <= 32, however)
                    constexpr unsigned pow2_divisor_N = pow2_divisor(impl_N);
                    constexpr unsigned num_sub_warp = (pow2_divisor_N > 32u) ? 32 : pow2_divisor_N;
                    constexpr unsigned sub_warp_NT = 32 / num_sub_warp; 
                    constexpr unsigned sub_warp_N   = impl_N / num_sub_warp;

                    unsigned j  = sub_warp_N * (lane_id / sub_warp_NT);
                    T*       bj = &index<T>(B, ldb, 0, j, trans_b, ArrangeB);

                    constexpr unsigned nrows = (impl_M + sub_warp_NT - 1) / sub_warp_NT;

                    T rmem1[nrows * sub_warp_N];
                    T rmem2[sub_warp_N];
                    T rmem3[impl_M];
                    trsm::warp_impl<T>(A, lda, bj, ldb, lane_id, rmem1, rmem2, rmem3, sub_warp_NT, impl_M, sub_warp_N, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                }
            }

        } else if constexpr (2 * impl_N >= NT || impl_M < 8u) {
            // Large number of RHSs or tiny matrix
            if constexpr (impl_N % NT == 0) {
                // Can evenly divide RHSs to threads
                // This has less smem IO than the next branch if impl_N > NT
                constexpr unsigned thread_N = impl_N / NT;
                unsigned           j        = thread_N * thread_id;

                T* bj = &index<T>(B, ldb, 0, j, trans_b, ArrangeB);

                T rmem[impl_M * thread_N];
                trsm::thread_impl<T>(A, lda, bj, ldb, rmem, impl_M, thread_N, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
            } else {
                // Just distribute RHSs round robin
                for (int j = thread_id; j < int(impl_N); j += NT) {
                    T* bj = &index<T>(B, ldb, 0, j, trans_b, ArrangeB);

                    T rmem[impl_M];
                    trsm::thread_impl<T>(A, lda, bj, ldb, rmem, impl_M, 1, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
                }
            }

        } else {
            // If other heuristics fail, just use the whole CTA synchronously
            constexpr unsigned nrows = (impl_M + NT - 1) / NT;

            T rmem1[nrows * impl_N];
            T rmem2[impl_N];
            trsm::cta_impl<T>(A, lda, B, ldb, thread_id, rmem1, rmem2, impl_M, impl_N, NT, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
        }
    }

    // Wrapper for when batches are packed tightly
    template<class T, unsigned M, unsigned N, side Side, diag Diag, transpose Transpose, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned BPB>
    inline __device__ void block_execute(const T* A, const unsigned lda, T* B, const unsigned ldb, const unsigned thread_id) {
        const unsigned strideA = (Side == side::left) ? M * lda : N * lda;
        const unsigned strideB = (ArrangeB == arrangement::col_major) ? ldb * N : M * ldb;
        block_execute<T, M, N, Side, Diag, Transpose, Fill, ArrangeA, ArrangeB, NT, BPB>(A, lda, strideA, B, ldb, strideB, thread_id);
    }

    // thread execution
    // B is MxN, A is MxM if left side or NxN if right side
    template<class T, unsigned M, unsigned N, side Side, diag Diag, transpose Transpose, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB>
    inline __device__ void thread_execute(const T* A, const unsigned lda, T* B, const unsigned ldb) {
        constexpr bool is_left   = (Side == side::left);
        constexpr unsigned impl_M    = is_left ? M : N;
        constexpr unsigned impl_N    = is_left ? N : M;
        
        constexpr bool     unit_diag = (Diag == diag::unit);
        constexpr bool     conj_a    = Transpose == transpose::conj_transposed;
        constexpr bool     trans_b   = !is_left;
        constexpr bool     fwd       = !is_left != (Transpose != transpose::non_transposed) != (Fill == fill_mode::lower);

        T rmem[impl_M * impl_N];
        trsm::thread_impl<T>(A, lda, B, ldb, rmem, impl_M, impl_N, unit_diag, conj_a, trans_b, Fill, ArrangeA, ArrangeB, fwd);
    }


} // namespace cusolverdx::detail::trsm


#endif // CUSOLVERDX_DATABASE_TRSM_CUH
