// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_POTRF_CLUSTER_IMPL_CUH
#define CUSOLVERDX_DATABASE_POTRF_CLUSTER_IMPL_CUH

#include <cooperative_groups.h>
#include "cublasdx.hpp"

#include "cusolverdx/database/potrf_cluster_io.cuh"
#include "cusolverdx/database/trsm.cuh"
#include "cusolverdx/operators/enums.hpp"
#include "cusolverdx/types.hpp"
#include <type_traits>

#ifdef CUSOLVERDX_NO_COMMONDX_COMPLEX
    #error "Invalid combination of headers"
#endif
#include "cusolverdx/database/util.cuh" // is_real_v, real_type_t, scalar, is_stored_triangle


namespace cusolverdx::detail::potrf {

    // forward declaration of block_execute
    template<class T, unsigned N, fill_mode Fill, arrangement Arrange, unsigned NT, unsigned BPB, int Arch>
    inline __device__ void block_execute(T* A, const unsigned lda, int* info, const unsigned thread_id);

    //=========================================
    // cluster impl: full-DSMEM blocked Cholesky right-looking algorithm
    // The complete stored triangle is expected to be resident in distributed shared memory. Each CTA owns a
    // round-robin subset of triangular tiles, and all POTRF/TRSM/GEMM updates operate on DSMEM tile pointers.
    //=========================================
    template<class T,
             unsigned    N,
             fill_mode   Fill,
             arrangement Arrange,
             unsigned    NT,
             unsigned    TileSize,
             unsigned    BPC,
             unsigned    Arch>
    __device__ void cluster_impl(T* A, int* info, const unsigned thread_id) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        namespace cg               = cooperative_groups;
        cg::cluster_group cluster  = cg::this_cluster();
        const unsigned    block_id = cluster.block_rank();

        constexpr unsigned n_tiles_per_side = (N + TileSize - 1) / TileSize;
        __shared__ int     local_info;

        static_assert(BPC <= max_cluster_blocks);
        cusolverdx::byte* dsmem[BPC];
        auto*             tile_base = reinterpret_cast<cusolverdx::byte*>(A);
        for (unsigned i = 0; i < BPC; ++i) {
            dsmem[i] = cluster.map_shared_rank(tile_base, i);
        }

        // Define the GEMM description here for the trailing matrix update
        using GEMM_Arrange = std::conditional_t<
            Fill == fill_mode::lower,
            std::conditional_t<Arrange == arrangement::col_major,
                               cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::col_major>,
                               cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::row_major>>,
            std::conditional_t<Arrange == arrangement::col_major,
                               cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::col_major>,
                               cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>>>;
        using precision_type     = real_type_t<T>;
        constexpr auto gemm_type = is_real_v<T> ? cublasdx::type::real : cublasdx::type::complex;
        using GEMM               = cublasdx::detail::make_description_t<cublasdx::Size<TileSize, TileSize, TileSize>,
                                                          GEMM_Arrange,
                                                          cublasdx::Alignment<alignof(T), alignof(T), alignof(T)>,
                                                          cublasdx::Precision<precision_type>,
                                                          cublasdx::Type<gemm_type>,
                                                          cublasdx::Function<cublasdx::function::MM>,
                                                          cublasdx::LeadingDimension<TileSize, TileSize, TileSize>,
                                                          cublasdx::Block,
                                                          cublasdx::BlockDim<NT>,
                                                          cublasdx::SM<Arch>>;
        using cublasdx_type      = typename convert_to_commondx_type<T>::type;
        using ALoadOp = std::conditional_t<!is_real_v<T> && Fill == fill_mode::upper, cublasdx::conjugate, cublasdx::identity>;
        using BLoadOp = std::conditional_t<!is_real_v<T> && Fill == fill_mode::lower, cublasdx::conjugate, cublasdx::identity>;


        auto tile_smem_ptr = [&](const unsigned row, const unsigned col) {
            return stored_tile_ptr<T, TileSize, Fill, BPC>(dsmem, row, col);
        };

        auto tile_owner = [](const unsigned row, const unsigned col) { return stored_tile_owner<Fill, BPC>(row, col); };

        if (thread_id == 0 && block_id == 0) {
            *info = 0;
        }
        cluster.sync();

        for (unsigned k = 0; k < n_tiles_per_side; ++k) {
            T*             Akk      = tile_smem_ptr(k, k);
            const unsigned owner_kk = tile_owner(k, k);

            if (block_id == owner_kk) {
                if (thread_id == 0) {
                    local_info = 0;
                }
                __syncthreads();
                potrf::block_execute<T, TileSize, Fill, Arrange, NT, 1, Arch>(Akk, TileSize, &local_info, thread_id);
            }

            if (thread_id == 0 && block_id == owner_kk && local_info != 0) {
                *info = local_info + k * TileSize;
            }
            cluster.sync();

            if (k + 1 == n_tiles_per_side) {
                break;
            }

            if constexpr (Fill == fill_mode::upper) {
                for (unsigned j = k + 1; j < n_tiles_per_side; ++j) {
                    T*             Akj      = tile_smem_ptr(k, j);
                    const unsigned owner_kj = tile_owner(k, j);

                    if (block_id == owner_kj) {
                        trsm::block_execute<T,
                                            TileSize,
                                            TileSize,
                                            side::left,
                                            diag::non_unit,
                                            is_real_v<T> ? transpose::transposed : transpose::conj_transposed,
                                            Fill,
                                            Arrange,
                                            Arrange,
                                            NT,
                                            1>(Akk, TileSize, Akj, TileSize, thread_id);
                    }
                }
                cluster.sync();

                for (unsigned j = k + 1; j < n_tiles_per_side; ++j) {
                    T* Akj = tile_smem_ptr(k, j);

                    for (unsigned i = k + 1; i <= j; ++i) {
                        T*             Aki      = tile_smem_ptr(k, i);
                        T*             Aij      = tile_smem_ptr(i, j);
                        const unsigned owner_ij = tile_owner(i, j);

                        if (block_id == owner_ij) {
                            GEMM().execute(scalar<cublasdx_type>(-1.0),
                                           reinterpret_cast<const cublasdx_type*>(Aki),
                                           reinterpret_cast<const cublasdx_type*>(Akj),
                                           scalar<cublasdx_type>(1.0),
                                           reinterpret_cast<cublasdx_type*>(Aij),
                                           ALoadOp(),
                                           BLoadOp());
                        }
                    }
                }
            } else {
                for (unsigned j = k + 1; j < n_tiles_per_side; ++j) {
                    T*             Ajk      = tile_smem_ptr(j, k);
                    const unsigned owner_jk = tile_owner(j, k);

                    if (block_id == owner_jk) {
                        trsm::block_execute<T,
                                            TileSize,
                                            TileSize,
                                            side::right,
                                            diag::non_unit,
                                            is_real_v<T> ? transpose::transposed : transpose::conj_transposed,
                                            Fill,
                                            Arrange,
                                            Arrange,
                                            NT,
                                            1>(Akk, TileSize, Ajk, TileSize, thread_id);
                    }
                }
                cluster.sync();

                for (unsigned j = k + 1; j < n_tiles_per_side; ++j) {
                    T* Ajk = tile_smem_ptr(j, k);

                    for (unsigned i = j; i < n_tiles_per_side; ++i) {
                        T*             Aik      = tile_smem_ptr(i, k);
                        T*             Aij      = tile_smem_ptr(i, j);
                        const unsigned owner_ij = tile_owner(i, j);

                        if (block_id == owner_ij) {
                            GEMM().execute(scalar<cublasdx_type>(-1.0),
                                           reinterpret_cast<const cublasdx_type*>(Aik),
                                           reinterpret_cast<const cublasdx_type*>(Ajk),
                                           scalar<cublasdx_type>(1.0),
                                           reinterpret_cast<cublasdx_type*>(Aij),
                                           ALoadOp(),
                                           BLoadOp());
                        }
                    }
                }
            }
            __syncthreads();
        }
#else
        (void)A;
        (void)info;
        (void)thread_id;
#endif // __CUDA_ARCH__ >= 900
    }

} // namespace cusolverdx::detail::potrf
#endif // CUSOLVERDX_DATABASE_POTRF_CLUSTER_IMPL_CUH
