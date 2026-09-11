// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_BDSVD_CUH
#define CUSOLVERDX_DATABASE_BDSVD_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/bdsvd_db.cuh"
#include "cusolverdx/detail/block_execution_bugs.hpp"

namespace cusolverdx::detail::bdsvd {

    template<class T, class S>
    __device__ void thread_driver(T* d, T* e, S* U, int ldu, int U_batch_size, S* VT, int ldvt, int VT_batch_size,
                                  int* info, const unsigned tid,
                                  const unsigned N, const unsigned M_U, const unsigned M_V,
                                  const bool overwriteU, const bool overwriteVT,
                                  const arrangement arrangeU, const arrangement arrangeVT,
                                  const unsigned NT, const unsigned BPB);
    template<class T, class S>
    __device__ void warp_driver(  T* d, T* e, S* U, int ldu, int U_batch_size, S* VT, int ldvt, int VT_batch_size,
                                  int* info, const unsigned tid,
                                  const unsigned N, const unsigned M_U, const unsigned M_V,
                                  const bool overwriteU, const bool overwriteVT,
                                  const arrangement arrangeU, const arrangement arrangeVT,
                                  const unsigned NT, const unsigned BPB);

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, job JobU, job JobVT, int Arch) {
        auto thresholds = suggested_bpb_size_thresholds(T, JobU != job::no_vectors || JobVT != job::no_vectors, Arch);
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

    constexpr inline __device__ __host__ dim3 suggested_block_dim([[maybe_unused]] type_enum T, [[maybe_unused]] unsigned N, [[maybe_unused]] job JobU, [[maybe_unused]] job JobVT, unsigned BPB, [[maybe_unused]] int Arch) {
        if (BPB > 1u) {
            return ((BPB+32-1)/32)*32;
        } else {
            return 32;
        }
    }

    // Block execution with configs useful for gesvd
    template<class T, class S, unsigned N, unsigned M_U, unsigned M_V, bool overwriteU, bool overwriteVT, arrangement arrangeU, arrangement arrangeVT,
             unsigned NT, unsigned BPB, int Arch>
    __device__ void block_execute(T* d, T* e, S* U, int ldu, int U_batch_size, S* VT, int ldvt, int VT_batch_size, int* info, unsigned tid) {

        static_assert(is_real_v<T>, "BDSVD does not support complex A.");

        if ((M_U >= 16u || M_V >= 16u) && NT >= 32u && NT % 32u == 0 && 4 * BPB < NT ) { // TODO tune
            warp_driver<T, S>(d, e, U, ldu, U_batch_size, VT, ldvt, VT_batch_size, info, tid, N, M_U, M_V, overwriteU, overwriteVT, arrangeU, arrangeVT, NT, BPB);
        } else {
            thread_driver<T, S>(d, e, U, ldu, U_batch_size, VT, ldvt, VT_batch_size, info, tid, N, M_U, M_V, overwriteU, overwriteVT, arrangeU, arrangeVT, NT, BPB);
        }
        __syncthreads();
    }

    // Block execution with default configs
    template<class T, class S, unsigned N, job JobU, job JobVT, arrangement arrangeU, arrangement arrangeVT, unsigned NT, unsigned BPB, int Arch>
    __device__ void block_execute(T* d, T* e, S* U, int ldu, S* VT, int ldvt, int* info, unsigned tid) {
        CUSOLVERDX_NVBUG_5986343_BDSVD_HANDLER(Arch, is_real_v<T>, JobU, JobVT, N, NT);

        static_assert(JobU != job::overwrite_vectors && JobVT != job::overwrite_vectors, "BDSVD doesn't support job::overwrite_vectors.");

        constexpr unsigned M_U           = (JobU == job::no_vectors) ? 0 : N;
        constexpr unsigned M_V           = (JobVT == job::no_vectors) ? 0 : N;
        constexpr bool     overwriteU    = (JobU != job::multiply_vectors);
        constexpr bool     overwriteVT   = (JobVT != job::multiply_vectors);
        const unsigned     U_batch_size  = ldu * M_U;
        const unsigned     VT_batch_size = ldvt * M_V;

        block_execute<T, S, N, M_U, M_V, overwriteU, overwriteVT, arrangeU, arrangeVT, NT, BPB, Arch>(d, e, U, ldu, U_batch_size, VT, ldvt, VT_batch_size, info, tid);
    }

    // Thread execution with configs useful for gesvd
    template<class T, class S, unsigned N, unsigned M_U, unsigned M_V, bool overwriteU, bool overwriteVT, arrangement arrangeU, arrangement arrangeVT>
    __device__ void thread_execute(T* d, T* e, S* U, int ldu, int U_batch_size, S* VT, int ldvt, int VT_batch_size, int* info) {

        static_assert(is_real_v<T>, "BDSVD does not support complex A.");

        thread_driver<T, S>(d, e, U, ldu, U_batch_size, VT, ldvt, VT_batch_size, info, 0, N, M_U, M_V, overwriteU, overwriteVT, arrangeU, arrangeVT, 1, 1);
    }

    // Thread execution with default configs 
    template<class T, class S, unsigned N, job JobU, job JobVT, arrangement arrangeU, arrangement arrangeVT>
    __device__ void thread_execute(T* d, T* e, S* U, int ldu, S* VT, int ldvt, int* info) {

        static_assert(JobU != job::overwrite_vectors && JobVT != job::overwrite_vectors, "BDSVD doesn't support job::overwrite_vectors.");

        constexpr unsigned M_U           = (JobU == job::no_vectors) ? 0 : N;
        constexpr unsigned M_V           = (JobVT == job::no_vectors) ? 0 : N;
        constexpr bool     overwriteU    = (JobU != job::multiply_vectors);
        constexpr bool     overwriteVT   = (JobVT != job::multiply_vectors);
        const unsigned     U_batch_size  = ldu * M_U;
        const unsigned     VT_batch_size = ldvt * M_V;

        thread_execute<T, S, N, M_U, M_V, overwriteU, overwriteVT, arrangeU, arrangeVT>(d, e, U, ldu, U_batch_size, VT, ldvt, VT_batch_size, info);
    }

} // namespace cusolverdx::detail::bdsvd

#endif // CUSOLVERDX_DATABASE_BDSVD_CUH
