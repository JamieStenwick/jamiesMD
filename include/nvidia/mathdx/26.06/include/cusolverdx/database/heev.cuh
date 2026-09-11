// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HEEV_CUH
#define CUSOLVERDX_DATABASE_HEEV_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/detail/type_enum.hpp"

#include "cusolverdx/database/hetrd.cuh"
#include "cusolverdx/database/htev.cuh"
#include "cusolverdx/database/ungqr.cuh"
#include "cusolverdx/database/util.cuh"

#include "cusolverdx/database/heev_db.cuh"

namespace cusolverdx::detail::heev {

    template<class T>
    __device__ void repack_tridiag(const T* A, int lda, int batch_stride, real_type_t<T>* d, real_type_t<T>* e, unsigned tid, unsigned N, const fill_mode Fill, const arrangement Arrange, unsigned NT, unsigned BPB);
    template<class T>
    __device__ void ungtr_setup(T* A, int lda, const unsigned tid, const unsigned N, const fill_mode Fill, const arrangement arrangeV, const unsigned NT, const unsigned BPB);

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, job Job, int Arch) {
        auto thresholds = suggested_bpb_size_thresholds(T, Job != job::no_vectors, Arch);
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

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, job Job, unsigned BPB, int Arch) {
        if (BPB > 1u) {
            // TODO review this choice
            return ((BPB+32-1)/32)*32;
        } else {
            auto thresholds = suggested_block_dim_size_thresholds(T, Job != job::no_vectors, Arch);
            if (N <= thresholds[0]) {
                return 32;
            } else if (N <= thresholds[1]) {
                return 64;
            } else if (N <= thresholds[2]) {
                return 128;
            } else if (N <= thresholds[3]) {
                return 256;
            } else if (N <= thresholds[4]) {
                return 512;
            } else {
                return 1024;
            }
        }
    }

    // Number of words of type T needed as workspace
    constexpr inline __device__ __host__ int workspace_size(type_enum T, unsigned N, job jobV, unsigned NT, unsigned BPB, int Arch) {
        // Each batch needs N-1 words for tau (reused for e), plus hetrd needs a little extra workspace in some cases
        int tau_size = (N-1)*BPB;
        int hetrd_workspace = hetrd::workspace_size(T, N, NT, BPB, Arch);
        int e_size = (N-1) * BPB;

        if (jobV == job::no_vectors) {
            return const_max(tau_size, e_size) + hetrd_workspace;
        } else {
            return tau_size + const_max(hetrd_workspace, e_size);
        }
    }
    constexpr inline __device__ __host__ int workspace_size_thread([[maybe_unused]] type_enum T, unsigned N, job jobV) {
        // Each batch needs N-1 words for tau (reused for e)
        return (jobV == job::no_vectors) ? (N-1) : (N-1) * 2;
    }

    template<class T, unsigned N, fill_mode Fill, arrangement Arrange, job jobV, unsigned NT, unsigned BPB, int Arch, class real_t = real_type_t<T>>
    __device__ void block_execute(T* A, int lda, real_type_t<T>* lambda, T* workspace, int* info, unsigned tid) {

        static_assert(jobV == job::no_vectors || jobV == job::overwrite_vectors, "HEEV only supports job::no_vectors and job::overwrite_vectors.");

        // Reduce to tridiagonal
        T* tau = workspace;
        T* swork = tau + (N-1)*BPB;
        hetrd::block_execute<T, N, Fill, Arrange, NT, BPB, Arch>(A, lda, tau, swork, tid);

        // Repack tridiagonal matrix
        __syncthreads();
        real_t* d = lambda;
        real_t* e = reinterpret_cast<real_t*>(workspace + (jobV == job::overwrite_vectors ? (N-1)*BPB : 0));
        repack_tridiag(A, lda, N*lda, d, e, tid, N, Fill, Arrange, NT, BPB);
        __syncthreads();

        if constexpr (jobV == job::overwrite_vectors) {
            ungtr_setup(A, lda, tid, N, Fill, Arrange, NT, BPB);
            __syncthreads();

            ungqr::block_execute<T, N-1, N-1, N-1, Arrange, NT, BPB>(A + (1 + lda), lda, N*lda, tau, N-1, tid);
            __syncthreads();
        }

        constexpr auto htev_job = jobV == job::no_vectors ? job::no_vectors : job::multiply_vectors;

        // Solve for values
        htev::block_execute<real_t, T, N, htev_job, Arrange, NT, BPB, Arch>(d, e, A, lda, info, tid);
    }
    
    template<class T, unsigned N, fill_mode Fill, arrangement Arrange, job jobV, class real_t = real_type_t<T>>
    __device__ void thread_execute(T* A, int lda, real_type_t<T>* lambda, T* workspace, int* info) {

        static_assert(jobV == job::no_vectors || jobV == job::overwrite_vectors, "HEEV only supports job::no_vectors and job::overwrite_vectors.");

        // Reduce to tridiagonal
        T* tau = workspace;
        hetrd::thread_execute<T, N, Fill, Arrange>(A, lda, tau);

        // Repack tridiagonal matrix
        real_t* d = lambda;
        real_t* e = reinterpret_cast<real_t*>(workspace + (jobV == job::overwrite_vectors ? (N-1) : 0));
        repack_tridiag(A, lda, N*lda, d, e, 0, N, Fill, Arrange, 1, 1);

        if constexpr (jobV == job::overwrite_vectors) {
            ungtr_setup(A, lda, 0, N, Fill, Arrange, 1, 1);

            ungqr::thread_execute<T, N-1, N-1, N-1, Arrange>(A + (1 + lda), lda, tau);
        }

        constexpr auto htev_job = jobV == job::no_vectors ? job::no_vectors : job::multiply_vectors;

        // Solve for values
        htev::thread_execute<real_t, T, N, htev_job, Arrange>(d, e, A, lda, info);

    }
} // namespace cusolverdx::detail::heev

#endif // CUSOLVERDX_DATABASE_HEEV_CUH
