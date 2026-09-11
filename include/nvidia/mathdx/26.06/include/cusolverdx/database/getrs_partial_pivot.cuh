// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_BATCH_GETRS_PARTIAL_PIVOT_CUH
#define CUSOLVERDX_DATABASE_BATCH_GETRS_PARTIAL_PIVOT_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/trsm.cuh"
#include "cusolverdx/database/indexing.cuh"

namespace cusolverdx::detail::getrs_partial_pivot {
    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, unsigned K, int Arch) {
        return trsm::suggested_batches(T, N, K, side::left, Arch);
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, unsigned K, unsigned BPB, int Arch) {
        return trsm::suggested_block_dim(T, N, K, side::left, BPB, Arch);
    }

    template<class T, unsigned N, unsigned K, arrangement ArrangeB, bool forward, unsigned NT, unsigned BPB>
    inline __device__ void laswp(T* B, const unsigned ldb, const int* ipiv, const unsigned thread_id) {
        // 1 thread per RHS
        for (int j = thread_id; j < int(BPB * K); j += NT) {
            unsigned batch = j / K;
            unsigned RHS   = j % K;

            auto Bj = B + batch * (ArrangeB == arrangement::col_major ? ldb * K : N * ldb);
            auto bj = &index(Bj, ldb, 0, RHS, ArrangeB);

            constexpr int i_start = forward ? 0 : N - 1;
            constexpr int i_inc   = forward ? 1 : -1;
            for (int i = i_start; (forward ? i < int(N) : i >= 0); i += i_inc) {
                int piv = ipiv[i + batch * N] - 1;
                if (i != piv) {
                    T temp                           = index(bj, ldb, i, 0, ArrangeB);
                    index(bj, ldb, i, 0, ArrangeB)   = index(bj, ldb, piv, 0, ArrangeB);
                    index(bj, ldb, piv, 0, ArrangeB) = temp;
                }
            }
        }
    }

    template<class T, unsigned N, unsigned K, arrangement ArrangeA, arrangement ArrangeB, transpose Trans, unsigned NT, unsigned BPB>
    inline __device__ void block_execute(const T* A, const unsigned lda, const int* ipiv, T* B, const unsigned ldb, const unsigned thread_id) {

        if constexpr (Trans == non_trans) {
            laswp<T, N, K, ArrangeB, true, NT, BPB>(B, ldb, ipiv, thread_id);
            __syncthreads();
        }

        if constexpr (Trans == non_trans) {
            trsm::block_execute<T, N, K, side::left, diag::unit, Trans, fill_mode::lower, ArrangeA, ArrangeB, NT, BPB>(A, lda, B, ldb, thread_id);
            trsm::block_execute<T, N, K, side::left, diag::non_unit, Trans, fill_mode::upper, ArrangeA, ArrangeB, NT, BPB>(A, lda, B, ldb, thread_id);
        } else {
            trsm::block_execute<T, N, K, side::left, diag::non_unit, Trans, fill_mode::upper, ArrangeA, ArrangeB, NT, BPB>(A, lda, B, ldb, thread_id);
            trsm::block_execute<T, N, K, side::left, diag::unit, Trans, fill_mode::lower, ArrangeA, ArrangeB, NT, BPB>(A, lda, B, ldb, thread_id);
        }

        if constexpr (Trans != non_trans) {
            __syncthreads();
            laswp<T, N, K, ArrangeB, false, NT, BPB>(B, ldb, ipiv, thread_id);
        }
    }

    template<class T, unsigned N, unsigned K, arrangement ArrangeA, arrangement ArrangeB, transpose Trans>
    inline __device__ void thread_execute(const T* A, const unsigned lda, const int* ipiv, T* B, const unsigned ldb) {

        if constexpr (Trans == non_trans) {
            laswp<T, N, K, ArrangeB, true, 1, 1>(B, ldb, ipiv, 0);
        }

        if constexpr (Trans == non_trans) {
            trsm::thread_execute<T, N, K, side::left, diag::unit, Trans, fill_mode::lower, ArrangeA, ArrangeB>(A, lda, B, ldb);
            trsm::thread_execute<T, N, K, side::left, diag::non_unit, Trans, fill_mode::upper, ArrangeA, ArrangeB>(A, lda, B, ldb);
        } else {
            trsm::thread_execute<T, N, K, side::left, diag::non_unit, Trans, fill_mode::upper, ArrangeA, ArrangeB>(A, lda, B, ldb);
            trsm::thread_execute<T, N, K, side::left, diag::unit, Trans, fill_mode::lower, ArrangeA, ArrangeB>(A, lda, B, ldb);
        }

        if constexpr (Trans != non_trans) {
            laswp<T, N, K, ArrangeB, false, 1, 1>(B, ldb, ipiv, 0);
        }
    }
} // namespace cusolverdx::detail::getrs_partial_pivot

#endif
