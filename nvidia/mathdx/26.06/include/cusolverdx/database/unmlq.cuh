// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_UNMLQ_CUH
#define CUSOLVERDX_DATABASE_UNMLQ_CUH

#include "cusolverdx/operators/enums.hpp"
#include "cusolverdx/database/unmqr.cuh"

namespace cusolverdx::detail::unmlq {
    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned M, unsigned N, int Arch) {
        return unmqr::suggested_batches(T, N, M, Arch);
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned M, unsigned N, unsigned BPB, int Arch) {
        return unmqr::suggested_block_dim(T, N, M, BPB, Arch);
    }

    template<class T, unsigned M, unsigned N, unsigned K, side Side, transpose Trans, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned BPB = 1>
    inline __device__ void block_execute(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id) {
        constexpr auto NewTrans = Trans == transpose::non_transposed ? transpose::conj_transposed : transpose::non_transposed;
        return unmqr::actual_dispatch<T, M, N, K, Side, NewTrans, ArrangeA, transpose::conj_transposed, ArrangeB, NT, BPB>(A, lda, tau, B, ldb, thread_id);
    }

    template<class T, unsigned M, unsigned N, unsigned K, side Side, transpose Trans, arrangement ArrangeA, arrangement ArrangeB>
    inline __device__ void thread_execute(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb) {
        constexpr auto NewTrans = Trans == transpose::non_transposed ? transpose::conj_transposed : transpose::non_transposed;
        constexpr unsigned BM = M;
        constexpr unsigned BN = N;
        constexpr unsigned AM = (Side == side::left) ? M : N;

        T rmem[AM + BM * BN];
        unmqr::thread_driver<T>(A, lda, tau, B, ldb, 0, M, N, K, Side, NewTrans, ArrangeA, transpose::conj_transposed, ArrangeB, 1, 1, rmem);
    }
} // namespace cusolverdx::detail::unmlq

#endif // CUSOLVERDX_DATABASE_UNMLQ_CUH
