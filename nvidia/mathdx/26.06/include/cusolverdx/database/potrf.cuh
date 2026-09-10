// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_POTRF_CUH
#define CUSOLVERDX_DATABASE_POTRF_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/indexing.cuh"
#include "cusolverdx/database/potrf_db.cuh"

namespace cusolverdx::detail::potrf {

    ////////// Cholesky Driver functions //////////
    template<class T>
    __device__ void thread_driver(T*                A,
                                  const unsigned    lda,
                                  int*              info,
                                  unsigned          thread_id,
                                  T*                rmem,
                                  const unsigned    N,
                                  const unsigned    NT,
                                  const unsigned    BPB,
                                  const fill_mode   Fill,
                                  const arrangement Arrange);

    template<class T>
    __device__ void partial_warp_driver(T*                A,
                                        const unsigned    lda,
                                        int*              info,
                                        unsigned          thread_id,
                                        T*                rmem1,
                                        T*                rmem2,
                                        T*                rmem3,
                                        const unsigned    N,
                                        const unsigned    NT,
                                        const unsigned    BPB,
                                        const unsigned    p,
                                        const unsigned    q,
                                        const fill_mode   Fill,
                                        const arrangement Arrange);

    template<class T>
    __device__ void warp_driver(T*                A,
                                const unsigned    lda,
                                int*              info,
                                unsigned          thread_id,
                                T*                rmem1,
                                T*                rmem2,
                                T*                rmem3,
                                const unsigned    N,
                                const unsigned    NT,
                                const unsigned    BPB,
                                const unsigned    p,
                                const unsigned    q,
                                const fill_mode   Fill,
                                const arrangement Arrange);

    template<class T>
    __device__ void cta_driver(T*                A,
                               const unsigned    lda,
                               int*              info,
                               unsigned          thread_id,
                               T*                rmem1,
                               T*                rmem2,
                               T*                rmem3,
                               const unsigned    N,
                               const unsigned    NT,
                               const unsigned    BPB,
                               const unsigned    p,
                               const unsigned    q,
                               const fill_mode   Fill,
                               const arrangement Arrange);

    ////////// thresholds for implementation dispatch ////////////
    template<class T, unsigned N, unsigned NT, unsigned BPB, int Arch>
    constexpr inline __device__ __host__ bool use_thread_per_batch() {
        // Check if matrix is small enough to always use thread per
        if (N <= tiny_threshold(type_to_enum<T>, Arch))
            return true;

        // If the number of batches is enough to saturate the threads, without excessive register strain
        // based on H100
        return BPB > (4u + sizeof(T)) * NT / 32u && N <= 8u;
    }

    template<class T, unsigned N, unsigned NT, unsigned BPB, int Arch>
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

        return N <= small_threshold(type_to_enum<T>, Arch);
    }

    template<class T, unsigned N, unsigned NT, unsigned BPB, int Arch>
    constexpr inline __device__ __host__ bool use_warp_per_batch() {
        constexpr bool even_warps = NT % 32 == 0;

        if (NT == 32) {
            return true;
        }

        return BPB > 1u && even_warps && NT >= 64u && N <= med_threshold(type_to_enum<T>, Arch);
    }


    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, int Arch) {
        auto thresholds = suggested_bpb_size_thresholds(T, Arch);
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

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, unsigned BPB, int Arch) {
        // Targets throughput bound cases

        if (BPB > 1u) {
            // The suggested batch counts all work out to prefer 1 warp when the suggestion is greater than 1.
            unsigned ideal_batches_per_warp = suggested_batches(T, N, Arch);
            bool     target_partial_warp    = ideal_batches_per_warp > 1u;

            if (N <= 8u) {
                return BPB <= 32u ? 32 : 64;

            } else if (target_partial_warp && BPB <= ideal_batches_per_warp) {
                return 32;

            } else if (target_partial_warp && BPB % ideal_batches_per_warp == 0) {
                return 32 * BPB / ideal_batches_per_warp;

            } else if (N <= (T != type_enum::complex_f64 ? 64u : 32u)) {
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
                return suggested_block_dim(T, N, 1, Arch);
            }
        } else {
            auto thresholds = suggested_block_dim_size_thresholds(T, Arch);
            if (N <= thresholds[0]) {
                return 32;
            } else if (N <= thresholds[1]) {
                return 64;
            } else if (N <= thresholds[2]) {
                return 128;
            } else if (N <= thresholds[3]) {
                return 256;
            } else if (N <= thresholds[4]){
                return 320;
            } else {
                return 256;
            }
        }
    }

    template<class T, unsigned N, fill_mode Fill, arrangement Arrange, unsigned NT, unsigned BPB, int Arch>
    inline __device__ void block_execute(T* A, const unsigned lda, int* info, const unsigned thread_id) {

        if constexpr (use_thread_per_batch<T, N, NT, BPB, Arch>()) {
            T rmem[N * N];

            thread_driver<T>(A, lda, info, thread_id, rmem, N, NT, BPB, Fill, Arrange);
            __syncthreads();

        } else if constexpr (use_partial_warp_per_batch<T, N, NT, BPB, Arch>()) {
            constexpr unsigned threads_per_batch = NT / BPB;

            constexpr unsigned p = threads_per_batch == 32u ? 8 : (threads_per_batch >= 8u ? 4 : 2);
            constexpr unsigned q = threads_per_batch >= 16u ? 4 : (threads_per_batch >= 4u ? 2 : 1);

            constexpr unsigned nrows = (N + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;

            // register memory
            T rmem1[nrows * ncols];
            T rmem2[nrows];
            T rmem3[ncols];

            partial_warp_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, N, NT, BPB, p, q, Fill, Arrange);
            __syncthreads();

            // If we only have 1 warp, just use the warp routine
        } else if constexpr (use_warp_per_batch<T, N, NT, BPB, Arch>()) {
            constexpr unsigned p = 8;
            constexpr unsigned q = 4;

            constexpr unsigned nrows = (N + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;

            // register memory
            T rmem1[nrows * ncols];
            T rmem2[nrows];
            T rmem3[ncols];

            warp_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, N, NT, BPB, p, q, Fill, Arrange);
            __syncthreads();

        } else {
            constexpr unsigned p = (NT % 16 != 0) ? 1 : (NT <= 64u ? 8 : 16);
            constexpr unsigned q = NT / p;

            constexpr unsigned nrows = (N + p - 1) / p;
            constexpr unsigned ncols = (N + q - 1) / q;

            static_assert(p * q == NT);
            static_assert(p <= 32u); // column bcasts utilize warp shuffle instructions

            // register memory
            T rmem1[nrows * ncols];
            T rmem2[nrows];
            T rmem3[ncols];

            cta_driver<T>(A, lda, info, thread_id, rmem1, rmem2, rmem3, N, NT, BPB, p, q, Fill, Arrange);
        }
    }

    template<class T, unsigned N, fill_mode Fill, arrangement Arrange>
    inline __device__ void thread_execute(T* A, const unsigned lda, int* info) {
        T rmem[N * N];

        thread_driver<T>(A, lda, info, 0, rmem, N, 1, 1, Fill, Arrange);
    }
} // namespace cusolverdx::detail::potrf


#endif // CUSOLVERDX_DATABASE_POTRF_CUH
