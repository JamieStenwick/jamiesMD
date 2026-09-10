// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_UNGQR_CUH
#define CUSOLVERDX_DATABASE_UNGQR_CUH

#include "cusolverdx/operators/enums.hpp"
#include "cusolverdx/database/geqrf.cuh"
#include "cusolverdx/database/ungqr_db.cuh"

namespace cusolverdx {
    namespace detail {
        namespace ungqr {

            template<class T>
            __device__ void thread_driver(T*                A,
                                          const unsigned    lda,
                                          const unsigned    batch_stride,
                                          const T*          tau,
                                          const unsigned    tau_stride,
                                          const unsigned    thread_id,
                                          const unsigned    M,
                                          const unsigned    N,
                                          const unsigned    K,
                                          const arrangement ArrangeA,
                                          const unsigned    NT,
                                          const unsigned    BPB,
                                          T*                rmem,
                                          const bool        need_conjugate_tau);
            template<class T>
            __device__ void warp_driver(T*                A,
                                        const unsigned    lda,
                                        const unsigned    batch_stride,
                                        const T*          tau,
                                        const unsigned    tau_stride,
                                        const unsigned    thread_id,
                                        const unsigned    M,
                                        const unsigned    N,
                                        const unsigned    K,
                                        const arrangement ArrangeA,
                                        const unsigned    NT,
                                        const unsigned    BPB,
                                        T*                rmem,
                                        const bool        need_conjugate_tau);
            template<class T>
            __device__ void warp_cta_driver(T*                A,
                                            const unsigned    lda,
                                            const unsigned    batch_stride,
                                            const T*          tau,
                                            const unsigned    tau_stride,
                                            const unsigned    thread_id,
                                            const unsigned    M,
                                            const unsigned    N,
                                            const unsigned    K,
                                            const arrangement ArrangeA,
                                            const unsigned    NT,
                                            const unsigned    BPB,
                                            T*                rmem,
                                            const bool        need_conjugate_tau);


            template<class T, unsigned M, unsigned N, unsigned K, arrangement ArrangeA, unsigned NT, unsigned BPB = 1>
            inline __device__ void block_execute(T* A, const unsigned lda, const unsigned batch_stride, const T* tau, const unsigned tau_stride, const unsigned thread_id, const bool need_conjugate_tau = false) {
                constexpr bool even_warps = NT % 32 == 0;

                if constexpr (M * N < 8u * 8u) {
                    T rmem[M * N + M]; // + M for the workspace of v vector
                    thread_driver<T>(A, lda, batch_stride, tau, tau_stride, thread_id, M, N, K, ArrangeA, NT, BPB, rmem, need_conjugate_tau);
                } else if constexpr (even_warps) {
                    T rmem[M * N + M + N];
                    warp_cta_driver<T>(A, lda, batch_stride, tau, tau_stride, thread_id, M, N, K, ArrangeA, NT, BPB, rmem, need_conjugate_tau);
                } else {
                    constexpr unsigned NT_used = (NT / 32) * 32;
                    if (thread_id < NT_used) {
                        T rmem[M * N + M + N];
                        warp_driver<T>(A, lda, batch_stride, tau, tau_stride, thread_id, M, N, K, ArrangeA, NT_used, BPB, rmem, need_conjugate_tau);
                    }
                }
                __syncthreads();
            }

            // Block execution with default strides
            template<class T, unsigned M, unsigned N, unsigned K, arrangement ArrangeA, unsigned NT, unsigned BPB = 1>
            inline __device__ void block_execute(T* A, const unsigned lda, const T* tau, const unsigned thread_id, const bool need_conjugate_tau = false) {
                const unsigned batch_stride = (ArrangeA == col_major) ? lda * N : M * lda;
                block_execute<T, M, N, K, ArrangeA, NT, BPB>(A, lda, batch_stride, tau, K, thread_id, need_conjugate_tau);
            }

            // Thread execution always use one batch per thread, no need to specify batch strides
            template<class T, unsigned M, unsigned N, unsigned K, arrangement ArrangeA>
            inline __device__ void thread_execute(T* A, const unsigned lda, const T* tau, const bool need_conjugate_tau = false) {
                T rmem[M * N + M]; // + M for the workspace of v vector
                thread_driver<T>(A, lda, 0, tau, 0, 0, M, N, K, ArrangeA, 1, 1, rmem, need_conjugate_tau);
            }

        } // namespace ungqr
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_UNGQR_CUH
