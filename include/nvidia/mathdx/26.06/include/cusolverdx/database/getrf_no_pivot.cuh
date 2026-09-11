// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GETRF_NO_PIVOT_CUH
#define CUSOLVERDX_DATABASE_GETRF_NO_PIVOT_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/getrf_no_pivot_db.cuh"

#include "cusolverdx/database/grid_utils.cuh"

namespace cusolverdx::detail::getrf_no_pivot {

    template<class T, bool is_modified_lu>
    __device__ void thread_driver(T* A, const unsigned lda, int* info, unsigned thread_id, T* rmem, unsigned M, unsigned N, unsigned NT, unsigned BPB, const arrangement Arrange, T* S = nullptr);

    template<class T, bool is_modified_lu>
    __device__ void partial_warp_driver(T*                A,
                                        const unsigned    lda,
                                        int*              info,
                                        unsigned          thread_id,
                                        T*                rmem1,
                                        T*                rmem2,
                                        T*                rmem3,
                                        unsigned          M,
                                        unsigned          N,
                                        unsigned          NT,
                                        unsigned          BPB,
                                        const unsigned    p,
                                        const unsigned    q,
                                        const arrangement Arrange,
                                        T*                S = nullptr);

    template<class T, bool is_modified_lu>
    __device__ void warp_driver(T*                A,
                                const unsigned    lda,
                                int*              info,
                                unsigned          thread_id,
                                T*                rmem1,
                                T*                rmem2,
                                T*                rmem3,
                                unsigned          M,
                                unsigned          N,
                                unsigned          NT,
                                unsigned          BPB,
                                const unsigned    p,
                                const unsigned    q,
                                const arrangement Arrange,
                                T*                S = nullptr);

    template<class T, bool is_modified_lu>
    __device__ void cta_driver(T*                A,
                               const unsigned    lda,
                               int*              info,
                               unsigned          thread_id,
                               T*                rmem1,
                               T*                rmem2,
                               T*                rmem3,
                               unsigned          M,
                               unsigned          N,
                               unsigned          NT,
                               unsigned          BPB,
                               const unsigned    p,
                               const unsigned    q,
                               const arrangement Arrange,
                               T*                S = nullptr);

    template<class T, unsigned M, unsigned N, unsigned NT, unsigned BPB, int Arch>
    constexpr inline __device__ __host__ bool use_thread_per_batch() {
        // Check if matrix is small enough to always use thread per
        constexpr unsigned tiny_thresh = tiny_threshold(type_to_enum<T>, Arch);
        if (M * N <= tiny_thresh)
            return true;

        // If the number of batches is enough to saturate the threads, without excessive register strain
        // Based on Cholesky tuning
        return BPB > (4u + sizeof(T)) * NT / 32u && M * N <= 64u;
    }

    template<class T, unsigned M, unsigned N, unsigned NT, unsigned BPB, int Arch>
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

        return M * N <= small_threshold(type_to_enum<T>, Arch);
    }

    template<class T, unsigned M, unsigned N, unsigned NT, unsigned BPB, int Arch>
    constexpr inline __device__ __host__ bool use_warp_per_batch() {
        constexpr bool even_warps = NT % 32 == 0;

        if (NT == 32) {
            return true;
        }

        return BPB > 1u && even_warps && NT >= 64u && M * N <= med_threshold(type_to_enum<T>, Arch);
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
            } else {
                return 512;
            }
        }
    }

    template<class T, unsigned M, unsigned N, arrangement Arrange, unsigned NT, unsigned BPB, int Arch, bool is_modified_lu = false>
    inline __device__ void block_execute(T* A, const unsigned lda, int* info, const unsigned thread_id, T* S = nullptr) {
        if constexpr (use_thread_per_batch<T, M, N, NT, BPB, Arch>()) {
            T rmem[M * N];

            thread_driver<T, is_modified_lu>(A, lda, info, thread_id, rmem, M, N, NT, BPB, Arrange, S);

        } else if constexpr (use_partial_warp_per_batch<T, M, N, NT, BPB, Arch>()) {
            static constexpr unsigned threads_per_batch = NT / BPB;
            static constexpr auto     pq                = pq_selector(M, N, threads_per_batch);
            static constexpr unsigned p                 = pq.p;
            static constexpr unsigned q                 = pq.q;

            constexpr unsigned nrows = (M + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;
            // register memory
            T rmem1[nrows * ncols];
            T rmem2[nrows];
            T rmem3[ncols];

            partial_warp_driver<T, is_modified_lu>(A, lda, info, thread_id, rmem1, rmem2, rmem3, M, N, NT, BPB, p, q, Arrange, S);

        } else if constexpr (use_warp_per_batch<T, M, N, NT, BPB, Arch>()) {
            static constexpr auto     pq = pq_selector(M, N, 32);
            static constexpr unsigned p  = pq.p;
            static constexpr unsigned q  = pq.q;

            constexpr unsigned nrows = (M + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;

            // register memory
            T rmem1[nrows * ncols];
            T rmem2[nrows];
            T rmem3[ncols];

            warp_driver<T, is_modified_lu>(A, lda, info, thread_id, rmem1, rmem2, rmem3, M, N, NT, BPB, p, q, Arrange, S);

        } else {
            static constexpr auto     pq = pq_selector(M, N, NT);
            static constexpr unsigned p  = pq.p;
            static constexpr unsigned q  = pq.q;

            constexpr unsigned nrows = (M + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;

            // register memory
            T rmem1[nrows * ncols];
            T rmem2[nrows];
            T rmem3[ncols];
            cta_driver<T, is_modified_lu>(A, lda, info, thread_id, rmem1, rmem2, rmem3, M, N, NT, BPB, p, q, Arrange, S);
        }

        __syncthreads();
    }

    template<class T, unsigned M, unsigned N, arrangement Arrange, bool is_modified_lu = false>
    inline __device__ void thread_execute(T* A, const unsigned lda, int* info, T* S = nullptr) {
        T rmem[M * N];

        thread_driver<T, is_modified_lu>(A, lda, info, 0, rmem, M, N, 1, 1, Arrange, S);
    }

} // namespace cusolverdx::detail::getrf_no_pivot

#endif //CUSOLVERDX_DATABASE_GETRF_NO_PIVOT_CUH
