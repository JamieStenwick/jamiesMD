// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_POTRF_CLUSTER_IO_CUH
#define CUSOLVERDX_DATABASE_POTRF_CLUSTER_IO_CUH

#include <cooperative_groups.h>

#include "cusolverdx/database/indexing.cuh"
#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/copy.hpp"
#include "cusolverdx/operators.hpp"
#include "cusolverdx/types.hpp"

namespace cusolverdx::detail::potrf {

    constexpr unsigned max_cluster_blocks = 16; // the size of DSMEM arrays needs to be compile-time constant

    // Get the address of the tile, with row/col tile-indices, in the matrix A
    template<class T, arrangement Arrange>
    inline __device__ T* matrix_tile(T* A, const unsigned lda, const unsigned tile_size, const unsigned row, const unsigned col) {
        if constexpr (Arrange == arrangement::col_major) {
            return A + row * tile_size + col * tile_size * lda;
        } else {
            return A + row * tile_size * lda + col * tile_size;
        }
    }

    template<fill_mode Fill>
    inline __device__ __host__ unsigned stored_tile_index(const unsigned row, const unsigned col) {
        if constexpr (Fill == fill_mode::lower) {
            return row * (row + 1) / 2 + col;
        } else {
            return col * (col + 1) / 2 + row;
        }
    }

    template<fill_mode Fill, unsigned BPC>
    inline __device__ __host__ unsigned stored_tile_owner(const unsigned row, const unsigned col) {
        return stored_tile_index<Fill>(row, col) % BPC;
    }

    template<fill_mode Fill, unsigned BPC>
    inline __device__ __host__ unsigned stored_tile_local_index(const unsigned row, const unsigned col) {
        return stored_tile_index<Fill>(row, col) / BPC;
    }

    template<class T, unsigned TileSize, fill_mode Fill, unsigned BPC>
    inline __device__ T* stored_tile_ptr(cusolverdx::byte** dsmem, const unsigned row, const unsigned col) {
        constexpr unsigned tile_elements = TileSize * TileSize;
        const unsigned     owner         = stored_tile_owner<Fill, BPC>(row, col);
        const unsigned     local_idx     = stored_tile_local_index<Fill, BPC>(row, col);
        return reinterpret_cast<T*>(dsmem[owner] + local_idx * tile_elements * sizeof(T));
    }

    // In case N % TileSize != 0
    template<unsigned N, unsigned TileSize>
    inline __device__ unsigned tile_extent(const unsigned tile_idx) {
        const unsigned start = tile_idx * TileSize;
        return (start + TileSize <= N) ? TileSize : (N - start);
    }

    template<class T, unsigned N, unsigned TileSize, unsigned NT, fill_mode Fill, arrangement Arrange>
    inline __device__ void load_tile(const T*       A,
                                     const unsigned lda,
                                     T*             tile,
                                     const unsigned tile_row,
                                     const unsigned tile_col,
                                     const bool     diagonal_tile,
                                     const bool     pad_identity,
                                     const unsigned thread_id) {
        __builtin_assume(thread_id < NT);

        const unsigned row_extent = tile_extent<N, TileSize>(tile_row);
        const unsigned col_extent = tile_extent<N, TileSize>(tile_col);

        if (row_extent == TileSize && col_extent == TileSize && !diagonal_tile) {
            copy_2d<NT, TileSize, TileSize, Arrange, 1>(A, lda, tile, TileSize);
            return;
        }

        // row major fast path for diagonal tile
        constexpr unsigned tile_elements = TileSize * TileSize;
        if constexpr (Arrange == arrangement::row_major) {
            if (row_extent == TileSize && col_extent == TileSize && diagonal_tile) {
                for (unsigned idx = thread_id; idx < tile_elements; idx += NT) {
                    const unsigned row = idx / TileSize;
                    const unsigned col = idx % TileSize;

                    T value {};
                    if (is_stored_triangle(row, col, Fill)) {
                        value = A[row * lda + col];
                    }
                    tile[idx] = value;
                }
                return;
            }
        }

        for (unsigned idx = thread_id; idx < tile_elements; idx += NT) {
            const unsigned row   = idx % TileSize;
            const unsigned col   = idx / TileSize;
            const bool     valid = row < row_extent && col < col_extent;

            T value {};
            if (valid) {
                if (!diagonal_tile || is_stored_triangle(row, col, Fill)) {
                    value = index_impl<T>(A, lda, row, col, false, Arrange);
                }
            } else if (pad_identity && row == col) {
                value = scalar<T>(1.0);
            }

            index_impl<T>(tile, TileSize, row, col, false, Arrange) = value;
        }
    }

    template<class T, unsigned N, unsigned TileSize, unsigned NT, fill_mode Fill, arrangement Arrange>
    inline __device__ void store_tile(const T*       tile,
                                      T*             A,
                                      const unsigned lda,
                                      const unsigned tile_row,
                                      const unsigned tile_col,
                                      const bool     diagonal_tile,
                                      const unsigned thread_id) {
        __builtin_assume(thread_id < NT);

        const unsigned row_extent = tile_extent<N, TileSize>(tile_row);
        const unsigned col_extent = tile_extent<N, TileSize>(tile_col);

        if (row_extent == TileSize && col_extent == TileSize && !diagonal_tile) {
            copy_2d<NT, TileSize, TileSize, Arrange, 1>(tile, TileSize, A, lda);
            return;
        }

        constexpr unsigned tile_elements = TileSize * TileSize;
        if constexpr (Arrange == arrangement::row_major) {
            if (row_extent == TileSize && col_extent == TileSize && diagonal_tile) {
                for (unsigned idx = thread_id; idx < tile_elements; idx += NT) {
                    const unsigned row = idx / TileSize;
                    const unsigned col = idx % TileSize;

                    if (is_stored_triangle(row, col, Fill)) {
                        A[row * lda + col] = tile[idx];
                    }
                }
                return;
            }
        }

        for (unsigned idx = thread_id; idx < tile_elements; idx += NT) {
            const unsigned row = idx % TileSize;
            const unsigned col = idx / TileSize;

            if (row < row_extent && col < col_extent && (!diagonal_tile || is_stored_triangle(row, col, Fill))) {
                index_impl<T>(A, lda, row, col, false, Arrange) = index_impl<T>(tile, TileSize, row, col, false, Arrange);
            }
        }
    }

    // Load diagonal tile from global memory to local shared memory
    template<class T, unsigned N, unsigned TileSize, unsigned NT, fill_mode Fill, arrangement Arrange>
    inline __device__ void load_diagonal_tile(const T*       A,
                                              const unsigned lda,
                                              T*             tile,
                                              const unsigned tile_idx,
                                              const unsigned thread_id) {
        load_tile<T, N, TileSize, NT, Fill, Arrange>(A, lda, tile, tile_idx, tile_idx, true, true, thread_id);
    }

    // Store diagonal tile from local shared memory to global memory
    template<class T, unsigned N, unsigned TileSize, unsigned NT, fill_mode Fill, arrangement Arrange>
    inline __device__ void store_diagonal_tile(const T*       tile,
                                               T*             A,
                                               const unsigned lda,
                                               const unsigned tile_idx,
                                               const unsigned thread_id) {
        store_tile<T, N, TileSize, NT, Fill, Arrange>(tile, A, lda, tile_idx, tile_idx, true, thread_id);
    }

    template<class T, unsigned N, unsigned TileSize, fill_mode Fill, arrangement Arrange, unsigned NT, unsigned BPC>
    inline __device__ void load_global_to_cluster_tiles(const T*       A,
                                                        const unsigned lda,
                                                        T*             cluster_tiles,
                                                        const unsigned thread_id) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        namespace cg               = cooperative_groups;
        cg::cluster_group cluster  = cg::this_cluster();
        const unsigned    block_id = cluster.block_rank();

        static_assert(BPC <= max_cluster_blocks);
        cusolverdx::byte* dsmem[BPC];
        auto*             tile_base = reinterpret_cast<cusolverdx::byte*>(cluster_tiles);
        for (unsigned i = 0; i < BPC; ++i) {
            dsmem[i] = cluster.map_shared_rank(tile_base, i);
        }

        constexpr unsigned n_tiles_per_side = (N + TileSize - 1) / TileSize;
        for (unsigned row = 0; row < n_tiles_per_side; ++row) {
            for (unsigned col = 0; col < n_tiles_per_side; ++col) {
                if (!is_stored_triangle(row, col, Fill) || block_id != stored_tile_owner<Fill, BPC>(row, col)) {
                    continue;
                }

                const T* tile_gmem = matrix_tile<T const, Arrange>(A, lda, TileSize, row, col);
                T*       tile_smem = stored_tile_ptr<T, TileSize, Fill, BPC>(dsmem, row, col);
                if (row == col) {
                    load_diagonal_tile<T, N, TileSize, NT, Fill, Arrange>(tile_gmem, lda, tile_smem, row, thread_id);
                } else {
                    load_tile<T, N, TileSize, NT, Fill, Arrange>(tile_gmem, lda, tile_smem, row, col, false, false, thread_id);
                }
            }
        }
        cluster.sync();
#else
        (void)A;
        (void)lda;
        (void)cluster_tiles;
        (void)thread_id;
#endif
    }

    template<class T, unsigned N, unsigned TileSize, fill_mode Fill, arrangement Arrange, unsigned NT, unsigned BPC>
    inline __device__ void store_cluster_tiles_to_global(T* A, const unsigned lda, T* cluster_tiles, const unsigned thread_id) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        namespace cg               = cooperative_groups;
        cg::cluster_group cluster  = cg::this_cluster();
        const unsigned    block_id = cluster.block_rank();

        cluster.sync();

        static_assert(BPC <= max_cluster_blocks);
        cusolverdx::byte* dsmem[BPC];
        auto*             tile_base = reinterpret_cast<cusolverdx::byte*>(cluster_tiles);
        for (unsigned i = 0; i < BPC; ++i) {
            dsmem[i] = cluster.map_shared_rank(tile_base, i);
        }

        constexpr unsigned n_tiles_per_side = (N + TileSize - 1) / TileSize;
        for (unsigned row = 0; row < n_tiles_per_side; ++row) {
            for (unsigned col = 0; col < n_tiles_per_side; ++col) {
                if (!is_stored_triangle(row, col, Fill) || block_id != stored_tile_owner<Fill, BPC>(row, col)) {
                    continue;
                }

                T* tile_gmem = matrix_tile<T, Arrange>(A, lda, TileSize, row, col);
                T* tile_smem = stored_tile_ptr<T, TileSize, Fill, BPC>(dsmem, row, col);
                if (row == col) {
                    store_diagonal_tile<T, N, TileSize, NT, Fill, Arrange>(tile_smem, tile_gmem, lda, row, thread_id);
                } else {
                    store_tile<T, N, TileSize, NT, Fill, Arrange>(tile_smem, tile_gmem, lda, row, col, false, thread_id);
                }
            }
        }
#else
        (void)A;
        (void)lda;
        (void)cluster_tiles;
        (void)thread_id;
#endif
    }

} // namespace cusolverdx::detail::potrf

#endif // CUSOLVERDX_DATABASE_POTRF_CLUSTER_IO_CUH
