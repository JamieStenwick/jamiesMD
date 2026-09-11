// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GTSV_NO_PIVOT_CUH
#define CUSOLVERDX_DATABASE_GTSV_NO_PIVOT_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/gtsv_no_pivot_db.cuh"
#include "cusolverdx/database/util.cuh"

namespace cusolverdx::detail::gtsv_no_pivot {
    template<class T>
    __device__ void thread_driver(const T* dl, const T* d, const T* du, T* b, int ldb, int* info, T* rwork, const unsigned thread_id, const unsigned N, const unsigned NRHS, const unsigned NT, const unsigned Batches, const arrangement Arrange);
    template<class T>
    __device__ void warp_driver(const T* dl, const T* d, const T* du, T* b, int ldb, int* info, T* rwork, const unsigned thread_id, const unsigned subwarp_size, const unsigned N, const unsigned NRHS, const unsigned NT, const unsigned Batches, const arrangement Arrange);

    constexpr inline __device__ __host__ unsigned suggested_batches_1rhs(type_enum T, unsigned N, int Arch) {
        auto thresholds = suggested_bpb_size_thresholds(T, Arch);

        if (N <= thresholds[0]) {
            return 32;
        } else if (N <= thresholds[1]) {
            return 16;
        } else if (N <= thresholds[2]) {
            return 8;
        } else if (N <= thresholds[3]) {
            return 4;
        } else if (N <= thresholds[4]) {
            return 2;
        } else {
            return 1;
        }
    }

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, unsigned NRHS, int Arch) {
        int batches_1rhs = suggested_batches_1rhs(T, N, Arch);
        int NRHS_correction = smallest_pow2_greater(NRHS);

        if (NRHS_correction >= batches_1rhs) {
            return 1;
        } else {
            return batches_1rhs / NRHS_correction;
        }
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, unsigned NRHS, unsigned BPB, int Arch) {
        auto bpb_per_warp = suggested_batches_1rhs(T, N, Arch);
        auto effective_bpb = NRHS * BPB;

        return 32 * ((effective_bpb - 1)/bpb_per_warp + 1);
    }

    template<class T, unsigned N, unsigned NRHS, unsigned BPB, unsigned NT>
    constexpr inline unsigned __device__ compute_subwarp_size() {
        static_assert(NT % 32 == 0, "warp-based implementation current assumes NT is a multiple of 32");
        static_assert(BPB < NT, "warp-based implementation assumes that there are more threads than batches");

        int num_warps = NT/32;
        constexpr unsigned max_B_per_warp = pow2_divisor(BPB);

        if (BPB <= num_warps) {
            return 32;
        } else {
            return 32 / max_B_per_warp;
        }
    }

    template<class T, unsigned N, unsigned NRHS, arrangement Arrange, unsigned NT, unsigned BPB, int Arch>
    __device__ void block_execute(const T* dl, const T* d, const T* du, T* b, int ldb, int* info, const unsigned thread_id) {
        constexpr bool whole_warps = (NT % 32 == 0);

        if constexpr (N < tiny_threshold(type_to_enum<T>, Arch) || !whole_warps || BPB > NT/2) {
            T rwork[N*NRHS + N];
            thread_driver<T>(dl, d, du, b, ldb, info, rwork, thread_id, N, NRHS, NT, BPB, Arrange);
        } else {
            // Using warp or partial warp driver

            constexpr unsigned subwarp_size = compute_subwarp_size<T, N, NRHS, BPB, NT>();
            constexpr int local_N = (N - 1) / subwarp_size + 1;

            T rwork[2*local_N*NRHS + 5*local_N + 2*NRHS + 4];
            warp_driver<T>(dl, d, du, b, ldb, info, rwork, thread_id, subwarp_size, N, NRHS, NT, BPB, Arrange);
        }
    }

    template<class T, unsigned N, unsigned NRHS, arrangement Arrange>
    __device__ void thread_execute(const T* dl, const T* d, const T* du, T* b, int ldb, int* info) {
        T rwork[N*NRHS + N];
        thread_driver<T>(dl, d, du, b, ldb, info, rwork, 0, N, NRHS, 1, 1, Arrange);
    }
} // namespace cusolverdx::detail::gtsv_no_pivot

#endif // CUSOLVERDX_DATABASE_GTSV_NO_PIVOT_CUH
