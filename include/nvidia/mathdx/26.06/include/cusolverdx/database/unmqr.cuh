// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_UNMQR_CUH
#define CUSOLVERDX_DATABASE_UNMQR_CUH

#include "cusolverdx/operators/enums.hpp"
#include "cusolverdx/database/unmqr_db.cuh"

namespace cusolverdx {
    namespace detail {

        namespace unmqr {
            // The Trans argument is the transposition of the mathematical Q requested by the user
            // The TransA argument is the transposition of the physical accesses of A.  This exists to handle QR and LQ with the same implementation

            template<class T>
            __device__ void thread_driver(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id, const unsigned M, const unsigned N, const unsigned K, const side Side, const transpose Trans, const arrangement ArrangeA, const transpose TransA, const arrangement ArrangeB, const unsigned NT, const unsigned BPB, T* rmem);
            template<class T>
            __device__ void partial_warp_driver(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id, const unsigned M, const unsigned N, const unsigned K, const side Side, const transpose Trans, const arrangement ArrangeA, const transpose TransA, const arrangement ArrangeB, const unsigned NT, const unsigned BPB, T* rmem);
            template<class T>
            __device__ void warp_driver(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id, const unsigned M, const unsigned N, const unsigned K, const side Side, const transpose Trans, const arrangement ArrangeA, const transpose TransA, const arrangement ArrangeB, const unsigned NT, const unsigned BPB, T* rmem);

    

            template<class T, unsigned M, unsigned N, unsigned K, side Side, transpose Trans, arrangement ArrangeA, transpose TransA, arrangement ArrangeB, unsigned NT, unsigned BPB=1>
            inline __device__ void actual_dispatch(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id) {

                constexpr bool even_warps = NT % 32 == 0;
                constexpr unsigned  num_warps  = NT / 32;

                constexpr unsigned BM = M;
                constexpr unsigned BN = N;
                constexpr unsigned AM = (Side == side::left) ? M : N;
                constexpr unsigned NRHS = (Side == side::left) ? N : M;

                constexpr unsigned threads_per_batch = NT / BPB;

                if constexpr (BPB > 1) {
                    // TODO need to work on these
                    if constexpr (BPB * K >= NT || AM < 8) {
                        // At least 1 RHS per thread or tiny matrices
                        T rmem[AM + BM * BN];
                        thread_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, BPB, rmem);
                    } else if constexpr (even_warps && BPB > num_warps && NT % BPB == 0 && 32 % threads_per_batch == 0) {
                        T rmem[BM * BN + BM + BN];
                        partial_warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, BPB, rmem);
                    } else if constexpr (even_warps) {
                        // At least 1 RHS per warp
                        static_assert(NT % 32 == 0);
                        T rmem[BM * BN + BM + BN];
                        warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, BPB, rmem);
                    } else {
                        // Irregular NT, only thread driver is reliable here
                        T rmem[AM + BM * BN];
                        thread_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, BPB, rmem);
                    }
                } else if constexpr (even_warps && NRHS <= NT && AM >= 8u) {
                    T rmem[BM * BN + BM + BN];
                    // this function falls back to warp_driver if certain conditions are not met
                    partial_warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, BPB, rmem);
                } else if constexpr (AM * K < 8 * 8) {  // tiny matrix use thread implementation
                    T rmem[AM + BM * BN];
                    thread_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT, BPB, rmem);
                } else { // otherwise, odd NT for example, or large number of RHSs, use warp implementation  
                    constexpr unsigned NT_used = (NT / 32) * 32;
                    if (thread_id < NT_used) {
                        T rmem[BM * BN + BM + BN];
                        partial_warp_driver<T>(A, lda, tau, B, ldb, thread_id, M, N, K, Side, Trans, ArrangeA, TransA, ArrangeB, NT_used, BPB, rmem);
                    }
                }
                __syncthreads();
            }

            template<class T, unsigned M, unsigned N, unsigned K, side Side, transpose Trans, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned BPB = 1>
            inline __device__ void block_execute(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb, const unsigned thread_id) {
                return actual_dispatch<T, M, N, K, Side, Trans, ArrangeA, transpose::non_transposed, ArrangeB, NT, BPB>(A, lda, tau, B, ldb, thread_id);
            }

            template<class T, unsigned M, unsigned N, unsigned K, side Side, transpose Trans, arrangement ArrangeA, arrangement ArrangeB>
            inline __device__ void thread_execute(const T* A, const unsigned lda, const T* tau, T* B, const unsigned ldb) {
                constexpr unsigned BM = M;
                constexpr unsigned BN = N;
                constexpr unsigned AM = (Side == side::left) ? M : N;

                T rmem[AM + BM * BN];
                thread_driver<T>(A, lda, tau, B, ldb, 0, M, N, K, Side, Trans, ArrangeA, transpose::non_transposed, ArrangeB, 1, 1, rmem);
            }
        } // namespace unmqr
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_UNMQR_CUH
