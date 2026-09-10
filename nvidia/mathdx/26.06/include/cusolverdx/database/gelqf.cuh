// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GELQF_CUH
#define CUSOLVERDX_DATABASE_GELQF_CUH

#include "cusolverdx/operators/enums.hpp"
#include "cusolverdx/detail/util.hpp"
#include "cusolverdx/database/geqrf.cuh"

namespace cusolverdx::detail::gelqf {
    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned M, unsigned N, int Arch) {
        return geqrf::suggested_batches(T, N, M, Arch);
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned M, unsigned N, unsigned BPB, int Arch) {
        return geqrf::suggested_block_dim(T, N, M, BPB, Arch);
    }

    template<class T, int M, int N, arrangement Arrange, int NT, unsigned BPB = 1>
    inline __device__ void block_execute(T* A, const unsigned lda, T* tau, const unsigned thread_id) {
        // GELQF is equivalent to doing GEQRF on the transpose, then conjugating tau
        // The arrangement is used to effect the transposition
        constexpr arrangement new_arr = (Arrange == col_major) ? row_major : col_major;
        geqrf::block_execute<T, N, M, new_arr, NT, BPB>(A, lda, tau, thread_id);
        if constexpr (!is_real_v<T>) {
            geqrf::conj_tau<T>(tau, const_min(M, N), thread_id, NT, BPB);
            __syncthreads();
        }
    }

    template<class T, int M, int N, arrangement Arrange>
    inline __device__ void thread_execute(T* A, const unsigned lda, T* tau) {
        constexpr arrangement new_arr = (Arrange == col_major) ? row_major : col_major;
        geqrf::thread_execute<T, N, M, new_arr>(A, lda, tau);
        if constexpr (!is_real_v<T>) {
            geqrf::conj_tau<T>(tau, const_min(M, N), 0, 1, 1);
        }
    }
} // namespace cusolverdx::detail::gelqf

#endif // CUSOLVERDX_DATABASE_GELQF_CUH
