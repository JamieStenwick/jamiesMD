// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GEQRF_CUH
#define CUSOLVERDX_DATABASE_GEQRF_CUH

#include "cusolverdx/operators/enums.hpp"
#include "cusolverdx/database/geqrf_db.cuh"
#include "cusolverdx/database/grid_utils.cuh"

namespace cusolverdx::detail {
    namespace geqrf {

        template<class T>       
        __device__ void thread_driver(T* A, const unsigned lda, T* tau, const unsigned thread_id, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned BPB);

        template<class T>
        __device__ void partial_warp_driver(T* A, const unsigned lda, T* tau, const unsigned thread_id, T* rmem, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned p, const unsigned q, const unsigned BPB);

        template<class T>
        __device__ void warp_driver(T* A, const unsigned lda, T* tau, const unsigned thread_id, T* rmem, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned p, const unsigned q, const unsigned BPB);

        template<class T>
        __device__ void cta_driver(T* A, const unsigned lda, T* tau, const unsigned thread_id, T* rmem, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned p, const unsigned q, const unsigned BPB);

        template<class T>
        __device__ void conj_tau(T* tau, const unsigned M, const unsigned thread_id, const unsigned NT, const unsigned Batches);

        // TODO tune size thresholds.  Current values are just guesses
        template<class T, unsigned M, unsigned N, unsigned NT, unsigned BPB>
        constexpr inline __device__ __host__ bool use_thread_per_batch() {
            // Other implementations require NT to be divisible by 32
            if (NT % 32 != 0) {
                return true;
            }

            return M < 4u || M * N < 4u * 4u;
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

            return M*N <= 24u*24u;
        }

        template<class T, unsigned M, unsigned N, unsigned NT, unsigned BPB>
        constexpr inline __device__ __host__ bool use_warp_per_batch() {
            constexpr bool even_warps = NT % 32 == 0;

            if (NT == 32) {
                return true;
            }

            return BPB > 1u && even_warps && NT >= 64u && M*N <= 64u*64u;
        }

        constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned M, unsigned N, int Arch) {
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


        constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned M, unsigned N, unsigned BPB, int Arch) {
            // Targets throughput bound cases

            if (BPB > 1u) {
                // The suggested batch counts all work out to prefer 1 warp when the suggestion is greater than 1.
                unsigned ideal_batches_per_warp = suggested_batches(T, M, N, Arch);
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
                } else if (M * N <= thresholds[3]) {
                    return 256;
                } else if (M * N <= thresholds[4]) {
                    return 512;
                } else {
                    return 1024;
                }
            }
        }

        template<class T, int M, int N, arrangement Arrange, int NT, unsigned BPB = 1>
        inline __device__ void block_execute(T* A, const unsigned lda, T* tau, const unsigned thread_id) {

            if constexpr (use_thread_per_batch<T, M, N, NT, BPB>()) {
                thread_driver<T>(A, lda, tau, thread_id, M, N, Arrange, NT, BPB);
            } else if constexpr (use_partial_warp_per_batch<T, M, N, NT, BPB>()) {
                T rmem[M * N + M + N]; // + M + N for the workspace

                constexpr unsigned threads_per_batch = NT / BPB;
                auto [p, q] = pq_selector(M, N, threads_per_batch);

                partial_warp_driver<T>(A, lda, tau, thread_id, rmem, M, N, Arrange, NT, p, q, BPB);
            } else if constexpr (use_warp_per_batch<T, M, N, NT, BPB>()) {
                T rmem[M * N + M + N]; // + M + N for the workspace
                auto [p, q] = pq_selector(M, N, 32);

                warp_driver<T>(A, lda, tau, thread_id, rmem, M, N, Arrange, NT, p, q, BPB);
            } else {
                static_assert(NT % 32 == 0);
                // Treat M as effectively larger due to asymmetries in the implementation
                constexpr auto raw_pq = pq_selector(1.5*M, N, NT);
                constexpr int p = raw_pq.p > 32u ? 32 : raw_pq.p;
                constexpr int q = NT / p;

                constexpr int nrows = (M - 1) / p + 1;
                constexpr int ncols = (N - 1) / q + 1;

                T rmem[nrows * ncols + nrows + ncols];

                cta_driver<T>(A, lda, tau, thread_id, rmem, M, N, Arrange, NT, p, q, BPB);
            }
            __syncthreads();
        }

        template<class T, int M, int N, arrangement Arrange>
        inline __device__ void thread_execute(T* A, const unsigned lda, T* tau) {
            thread_driver<T>(A, lda, tau, 0, M, N, Arrange, 1, 1);
        }

    } // namespace geqrf
} // namespace cusolverdx::detail

#endif // CUSOLVERDX_DATABASE_GEQRF_CUH
