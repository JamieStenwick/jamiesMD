// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GETRF_PARTIAL_PIVOT_CUH
#define CUSOLVERDX_DATABASE_GETRF_PARTIAL_PIVOT_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/getrf_partial_pivot_db.cuh"

#include "cusolverdx/database/grid_utils.cuh"

namespace cusolverdx::detail::getrf_partial_pivot {

    template<class T>
    __device__ void thread_driver(T*                A,
                                  const unsigned    lda,
                                  int*              ipiv,
                                  int*              info,
                                  const unsigned    thread_id,
                                  const unsigned    M,
                                  const unsigned    N,
                                  const arrangement Arrange,
                                  const unsigned    NT,
                                  const unsigned    BPB,
                                  T*                rmem);

    template<class T>
    __device__ void partial_warp_driver(T*                A,
                                        const unsigned    lda,
                                        int*              ipiv,
                                        int*              info,
                                        const unsigned    thread_id,
                                        const unsigned    M,
                                        const unsigned    N,
                                        const arrangement Arrange,
                                        const unsigned    NT,
                                        const unsigned    BPB,
                                        const unsigned    p,
                                        const unsigned    q,
                                        T*                A_col,
                                        T*                A_row);

    template<class T>
    __device__ void warp_driver(T*                A,
                                const unsigned    lda,
                                int*              ipiv,
                                int*              info,
                                const unsigned    thread_id,
                                const unsigned    M,
                                const unsigned    N,
                                const arrangement Arrange,
                                const unsigned    NT,
                                const unsigned    BPB,
                                const unsigned    p,
                                const unsigned    q,
                                T*                A_col,
                                T*                A_row);

    template<class T>
    __device__ void cta_driver(T*                A,
                               const unsigned    lda,
                               int*              ipiv,
                               int*              info,
                               const unsigned    thread_id,
                               const unsigned    M,
                               const unsigned    N,
                               const arrangement Arrange,
                               const unsigned    NT,
                               const unsigned    BPB,
                               const unsigned    p,
                               const unsigned    q,
                               T*                A_col,
                               T*                A_row);

    // TODO tune size thresholds.  Current values are just guesses
    template<class T, unsigned M, unsigned N, unsigned NT, unsigned BPB>
    constexpr inline __device__ __host__ bool use_thread_per_batch() {
        return M * N < 8u * 8u;
    }

    template<class T, unsigned M, unsigned N, unsigned NT, unsigned BPB>
    constexpr inline __device__ __host__ bool use_partial_warp_per_batch() {
        constexpr bool     even_warps        = NT % 32 == 0;
        constexpr unsigned threads_per_batch = NT / BPB;

        if (!even_warps || NT % BPB != 0 || 32 % threads_per_batch != 0) {
            // Can't align batches perfectly to partial warps
            return false;
        }
        if (threads_per_batch > 32u || threads_per_batch <= 1u) {
            // Other drivers are better
            return false;
        }

        return M * N <= 24u * 24u;
    }

    template<class T, unsigned M, unsigned N, unsigned NT, unsigned BPB>
    constexpr inline __device__ __host__ bool use_warp_per_batch() {
        constexpr bool even_warps = NT % 32 == 0;

        if (NT == 32) {
            return true;
        }

        return BPB > 1u && even_warps && NT >= 64u && M * N <= 64u * 64u;
    }

    constexpr inline __device__ __host__ unsigned suggested_batches_per_warp(type_enum T, unsigned M, unsigned N, int Arch) {
        auto thresholds = suggested_bpb_size_thresholds(T, Arch);
        if (M * N <= thresholds[0]) {
            return 32;
        } else if (M * N <= thresholds[1]) {
            return 16;
        } else if (M * N <= thresholds[2]) {
            return 8;
        } else if (M * N <= thresholds[3]) {
            return 4;
        } else if (M * N <= thresholds[4]) {
            return 2;
        } else {
            return 1;
        }
    }

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned M, unsigned N, int Arch) {
        auto per_warp = suggested_batches_per_warp(T, M, N, Arch);
        bool have_large_smem = (Arch == 800) || (Arch == 870) || (Arch == 900) || (Arch == 1000);
        if (per_warp != 1 && have_large_smem) {
            return 2 * per_warp;
        } else {
            return per_warp;
        }
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned M, unsigned N, unsigned BPB, int Arch) {
        // Targets throughput bound cases

        if (BPB > 1u) {
            unsigned ideal_batches_per_warp = suggested_batches_per_warp(T, M, N, Arch);
            bool     target_partial_warp    = ideal_batches_per_warp > 1u;

            if (M * N <= 8u) {
                return BPB <= 32u ? 32 : 64;

            } else if (target_partial_warp && BPB <= ideal_batches_per_warp) {
                return 32;

            } else if (target_partial_warp && BPB % ideal_batches_per_warp == 0) {
                return 32 * BPB / ideal_batches_per_warp;

            } else if (M * N <= (T != type_enum::complex_f64 ? 64u * 64u : 32u * 32u)) {
                if (BPB == 2) {
                    return 64;
                } else {
                    // Use 3 or 4 warps, depending on what results in fewer idle warps for the last wave of batches
                    unsigned rem_3 = (BPB % 3) ? 0 : 3 - (BPB % 3);
                    unsigned rem_4 = (BPB % 4) ? 0 : 4 - (BPB % 4);
                    return rem_3 < rem_4 ? 96 : 128;
                }
            } else {
                // For large sizes, just use suggestion for 1 batch per block.
                return suggested_block_dim(T, M, N, 1, Arch);
            }
        } else {
            auto thresholds = suggested_block_dim_size_thresholds(T, Arch);
            if (M * N <= thresholds[0]) {
                return 32;
            } else if (M * N <= thresholds[1]) {
                return 64;
            } else if (M * N <= thresholds[2]) {
                return 128;
            } else {
                return 256;
            }
        }
    }

    template<class T, unsigned M, unsigned N, arrangement Arrange, unsigned NT, unsigned BPB, int Arch>
    inline __device__ void block_execute(T* A, const unsigned lda, int* ipiv, int* info, const unsigned thread_id) {

        if constexpr (use_thread_per_batch<T, M, N, NT, BPB>()) {
            T rmem[N];
            thread_driver<T>(A, lda, ipiv, info, thread_id, M, N, Arrange, NT, BPB, rmem);
            __syncthreads();

        } else if constexpr (use_partial_warp_per_batch<T, M, N, NT, BPB>()) {
            constexpr unsigned threads_per_batch = NT / BPB;
            constexpr auto     pq                = pq_selector(M, N, threads_per_batch);
            constexpr unsigned p                 = pq.p;
            constexpr unsigned q                 = pq.q;
            static_assert(p * q == threads_per_batch);

            constexpr unsigned BatchesPerWarp = 32 / threads_per_batch;
            static_assert(BatchesPerWarp != 0);
            static_assert(BatchesPerWarp < 32u);
            static_assert(32 % BatchesPerWarp == 0);

            constexpr unsigned nrows = (M + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;

            T A_col_rmem[nrows];
            T A_row_rmem[ncols];

            partial_warp_driver<T>(A, lda, ipiv, info, thread_id, M, N, Arrange, NT, BPB, p, q, A_col_rmem, A_row_rmem);
            __syncthreads();

        } else if constexpr (use_warp_per_batch<T, M, N, NT, BPB>()) {
            constexpr auto     pq = pq_selector(M, N, 32);
            constexpr unsigned p  = pq.p;
            constexpr unsigned q  = pq.q;
            static_assert(p * q == 32);

            constexpr unsigned nrows = (M + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;

            T A_col_rmem[nrows];
            T A_row_rmem[ncols];

            warp_driver<T>(A, lda, ipiv, info, thread_id, M, N, Arrange, NT, BPB, p, q, A_col_rmem, A_row_rmem);
            __syncthreads();

        } else {
            constexpr auto     pq = pq_selector(M, N, NT);
            constexpr unsigned p  = pq.p;
            constexpr unsigned q  = pq.q;
            static_assert(p * q == NT);

            constexpr unsigned nrows = (M + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;

            T A_col_rmem[nrows];
            T A_row_rmem[ncols];

            cta_driver<T>(A, lda, ipiv, info, thread_id, M, N, Arrange, NT, BPB, p, q, A_col_rmem, A_row_rmem);
        }
    }

    template<class T, unsigned M, unsigned N, arrangement Arrange>
    inline __device__ void thread_execute(T* A, const unsigned lda, int* ipiv, int* info) {
        T rmem[N];
        thread_driver<T>(A, lda, ipiv, info, 0, M, N, Arrange, 1, 1, rmem);
    }
} // namespace cusolverdx::detail::getrf_partial_pivot

#endif // CUSOLVERDX_DATABASE_GETRF_PARTIAL_PIVOT_CUH
