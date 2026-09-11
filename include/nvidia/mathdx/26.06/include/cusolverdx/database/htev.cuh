// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HTEV_CUH
#define CUSOLVERDX_DATABASE_HTEV_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/htev_db.cuh"
#include "cusolverdx/detail/block_execution_bugs.hpp"

namespace cusolverdx::detail::htev {
    template<class T, class S>
    __device__ void thread_driver(T* d, T* e, S* v, int ldv, int* info, unsigned tid, S* rwork, unsigned N, job j, arrangement arrangeV, unsigned NT, unsigned BPB);
    template<class T, class S>
    __device__ void warp_driver(T* d, T* e, S* V, int ldv, int* info, unsigned tid, S* rwork, unsigned N, job jobV, arrangement arrangeV, unsigned NT, unsigned BPB);

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, job jobV, int Arch) {
        auto thresholds = suggested_bpb_size_thresholds(T, jobV != job::no_vectors, Arch);
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

    constexpr inline __device__ __host__ dim3 suggested_block_dim([[maybe_unused]] type_enum T, [[maybe_unused]] unsigned N, [[maybe_unused]] job jobV, unsigned BPB, [[maybe_unused]] int Arch) {
        if (BPB > 1u) {
            return ((BPB + 32 - 1) / 32) * 32;
        } else {
            return 32;
        }
    }

    template<class T, class S, unsigned N, job jobV, arrangement arrangeV, unsigned NT, unsigned BPB, int Arch>
    __device__ void block_execute(T* d, T* e, S* v, int ldv, int* info, unsigned tid) {

        static_assert(is_real_v<T>, "HTEV does not currently support complex A.");
        static_assert(jobV != job::overwrite_vectors, "HTEV doesn't support job::overwrite_vectors.");

        CUSOLVERDX_NVBUG_5972531_HTEV_HANDLER(Arch, is_real_v<T>, jobV, N, NT);

        if constexpr (jobV == job::no_vectors || N < 16u || 2 * BPB >= NT) {
            S rwork[2 * N];
            thread_driver<T>(d, e, v, ldv, info, tid, rwork, N, jobV, arrangeV, NT, BPB);
        } else {
            S rwork[2 * ((N - 1) / 32 + 1)];

            // skip extra threads if NT % 32 != 0
            int num_whole_warps = NT / 32;
            int whole_NT        = num_whole_warps * 32;
            if (NT % 32 == 0 || tid < whole_NT) {
                warp_driver<T>(d, e, v, ldv, info, tid, rwork, N, jobV, arrangeV, whole_NT, BPB);
            }
        }
        __syncthreads();
    }

    template<class T, class S, unsigned N, job jobV, arrangement arrangeV>
    __device__ void thread_execute(T* d, T* e, S* v, int ldv, int* info) {

        static_assert(is_real_v<T>, "HTEV does not currently support complex A.");
        static_assert(jobV != job::overwrite_vectors, "HTEV doesn't support job::overwrite_vectors.");

        S rwork[2 * N];
        thread_driver<T>(d, e, v, ldv, info, 0, rwork, N, jobV, arrangeV, 1, 1);
    }
} // namespace cusolverdx::detail::htev

#endif // CUSOLVERDX_DATABASE_HTEV_CUH
