// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_POTRF_CLUSTER_CUH
#define CUSOLVERDX_DATABASE_POTRF_CLUSTER_CUH

#include "commondx/device_info.hpp"

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/potrf.cuh"
#include "cusolverdx/database/potrf_cluster_db.cuh"
#include "cusolverdx/database/potrf_cluster_impl.cuh"

namespace cusolverdx::detail::potrf {

    template<class T, unsigned N, unsigned TileSize, unsigned BPC, unsigned Arch>
    constexpr inline __device__ __host__ unsigned shared_memory_size_cluster() {
        constexpr unsigned n_tiles_per_side = (N + TileSize - 1) / TileSize;
        constexpr unsigned num_tiles        = n_tiles_per_side * (n_tiles_per_side + 1) / 2;
        constexpr unsigned tiles_per_cta    = (num_tiles + BPC - 1) / BPC;
        return TileSize * TileSize * tiles_per_cta * sizeof(T);
    }

    template<class T, unsigned N, unsigned TileSize, unsigned BPC, unsigned Arch>
    constexpr inline __device__ __host__ bool potrf_cluster_fits_in_shared_memory() {
        constexpr size_t shared_memory_size = shared_memory_size_cluster<T, N, TileSize, BPC, Arch>() + sizeof(int); // local smem info slot
        return shared_memory_size <= commondx::device_info<Arch>::shared_memory();
    }

    template<class T, unsigned N, fill_mode Fill, arrangement Arrange, unsigned NT, unsigned TileSize, unsigned BPC, unsigned Arch>
    inline __device__ void cluster_execute(T* A, int* info, const unsigned thread_id) {
        static_assert(potrf_cluster_fits_in_shared_memory<T, N, TileSize, BPC, Arch>(),
                      "Cluster POTRF: the required full-DSMEM triangular tile storage does not fit in shared memory.");

        if constexpr (BPC == 1) {
            potrf::block_execute<T, N, Fill, Arrange, NT, 1, Arch>(A, N, info, thread_id);
        } else {
            cluster_impl<T, N, Fill, Arrange, NT, TileSize, BPC, Arch>(A, info, thread_id);
        }
    }
} // namespace cusolverdx::detail::potrf

#endif // CUSOLVERDX_DATABASE_POTRF_CLUSTER_CUH
