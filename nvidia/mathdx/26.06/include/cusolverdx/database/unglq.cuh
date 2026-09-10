// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_UNGLQ_CUH
#define CUSOLVERDX_DATABASE_UNGLQ_CUH

#include "cusolverdx/operators/enums.hpp"
#include "cusolverdx/database/ungqr.cuh"

namespace cusolverdx::detail::unglq {
        constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned M, unsigned N, int Arch) {
            return ungqr::suggested_batches(T, N, M, Arch);
        }

        constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned M, unsigned N, unsigned BPB, int Arch) {
            return ungqr::suggested_block_dim(T, N, M, BPB, Arch);
        }

        template<class T, unsigned M, unsigned N, unsigned K, arrangement ArrangeA, unsigned NT, unsigned BPB = 1>
        inline __device__ void block_execute(T* A, const unsigned lda, const unsigned batch_stride, const T* tau, const unsigned tau_stride, const unsigned thread_id) {
            constexpr arrangement new_arr = (ArrangeA == col_major) ? row_major : col_major;

            const bool need_conjugate_tau_unglq = true; 
            ungqr::block_execute<T, N, M, K, new_arr, NT, BPB>(A, lda, batch_stride, tau, tau_stride, thread_id, need_conjugate_tau_unglq);

        }

        // Block execution with default strides
        template<class T, unsigned M, unsigned N, unsigned K, arrangement ArrangeA, unsigned NT, unsigned BPB = 1>
        inline __device__ void block_execute(T* A, const unsigned lda, const T* tau, const unsigned thread_id) {
            const unsigned batch_stride = (ArrangeA == col_major) ? lda * N : M * lda;
            block_execute<T, M, N, K, ArrangeA, NT, BPB>(A, lda, batch_stride, tau, K, thread_id);
        }

        // Thread execution always use one batch per thread, no need to specify batch strides
        template<class T, unsigned M, unsigned N, unsigned K, arrangement ArrangeA>
        inline __device__ void thread_execute(T* A, const unsigned lda, const T* tau) {
            constexpr arrangement new_arr = (ArrangeA == col_major) ? row_major : col_major;

            const bool need_conjugate_tau_unglq = true;
            ungqr::thread_execute<T, N, M, K, new_arr>(A, lda, tau, need_conjugate_tau_unglq);
        }

} // namespace cusolverdx::detail::unglq

#endif // CUSOLVERDX_DATABASE_UNGLQ_CUH
