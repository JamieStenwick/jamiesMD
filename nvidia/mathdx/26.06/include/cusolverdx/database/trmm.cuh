// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software, or related documentation without an express
// license agreement from NVIDIA CORPORATION, is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_TRMM_CUH
#define CUSOLVERDX_DATABASE_TRMM_CUH

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/database/indexing.cuh"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx::detail::trmm {

    // Only Side=Left, Diag=NonUnit, alpha=1, implemented
    // Dispatch only setup for the M=N case

    template<class T>
    __device__ void thread_impl_left_nonunit(const T*          A, const unsigned    lda,
                                             T*                B, const unsigned    ldb,
                                             const unsigned    M, const unsigned    N,
                                             const transpose   Transpose,
                                             const fill_mode   Fill,
                                             const arrangement ArrangeA,
                                             const arrangement ArrangeB,
                                             T* rwork);
    template<class T>
    __device__ void warp_impl_left_nonunit(const T*          A, const unsigned    lda,
                                           T*                B, const unsigned    ldb,
                                           const unsigned    M, const unsigned    N,
                                           const transpose   Transpose,
                                           const fill_mode   Fill,
                                           const arrangement ArrangeA,
                                           const arrangement ArrangeB,
                                           const int tid, const int NT,
                                           T* rwork);

    template<class T, unsigned M, unsigned N, side Side, diag Diag, transpose Transpose, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned BPB>
    inline __device__ void block_execute(const T* A, const unsigned lda, T* B, const unsigned ldb, const unsigned thread_id) {
        constexpr bool is_left   = (Side == side::left);
        constexpr bool unit_diag = (Diag == diag::unit);

        static_assert(is_left && !unit_diag, "TRMM only supports key cases");
        static_assert((Fill == fill_mode::lower && Transpose == transpose::non_transposed) ||
                      (Fill == fill_mode::upper && Transpose != transpose::non_transposed), "TRMM only supports key cases");

        constexpr bool even_warps = NT % 32 == 0;
        constexpr int  num_warps  = NT / 32;

        const int strideA = is_left ? (M * lda) : (N * lda);
        const int strideB = ArrangeB == arrangement::col_major ? (ldb * N) : (M * ldb);

        if constexpr (BPB > 1u) {
            // Handle batched dispatch separately from non-batched

            // TODO try to do multiple RHSs together where possible
            if (BPB * N >= NT || M <= 8u || ! even_warps) {
                // At least 1 RHS per thread or tiny matrices
                for (int j = thread_id; j < int(BPB * N); j += NT) {
                    unsigned batch = j / N;
                    unsigned RHS   = j % N;

                    auto Aj = A + batch * strideA;
                    auto Bj = B + batch * strideB;
                    auto bj = &index<T>(Bj, ldb, 0, RHS, false, ArrangeB);

                    T rmem[M*M + M*1];

                    thread_impl_left_nonunit(Aj, lda, bj, ldb, M, 1, Transpose, Fill, ArrangeA, ArrangeB, rmem);
                }
            } else {
                // At least 1 RHS per warp
                unsigned warp_id = thread_id / 32;
                unsigned lane_id = thread_id % 32;
                for (int j = warp_id; j < int(BPB * N); j += num_warps) {
                    unsigned batch = j / N;
                    unsigned RHS   = j % N;

                    auto Aj = A + batch * strideA;
                    auto Bj = B + batch * strideB;
                    auto bj = &index<T>(Bj, ldb, 0, RHS, false, ArrangeB);

                    T rmem[M*M + 2*M*1];
                    warp_impl_left_nonunit(Aj, lda, bj, ldb, M, 1, Transpose, Fill, ArrangeA, ArrangeB, lane_id, 32, rmem);
                }
            }

        } else if constexpr (NT >= 32u && N < NT && M >= 8u) {
            // Have at least a warp, fewer RHSs than threads, and the matrix isn't tiny
            // Split the problem across warps
            if constexpr (even_warps && N % num_warps == 0 && N >= num_warps) {
                // Can split the RHS's nicely

                // NB, because N < NT, we have N_per_warp < 32
                constexpr unsigned N_per_warp = N / num_warps;
                // Compute the largest power of two for which N_per_warp is divisible, then use that many subwarps
                constexpr unsigned num_sub_warp = pow2_divisor(N_per_warp);
                static_assert(num_sub_warp == 1 || num_sub_warp == 2 || num_sub_warp == 4 || num_sub_warp == 8 || num_sub_warp == 16 || num_sub_warp == 32);
                static_assert(N_per_warp % num_sub_warp == 0);

                constexpr unsigned sub_warp_NT = 32 / num_sub_warp;
                constexpr unsigned sub_warp_N  = N_per_warp / num_sub_warp;

                unsigned sub_warp_id = thread_id / sub_warp_NT;
                unsigned lane_id     = thread_id % sub_warp_NT;

                unsigned j  = sub_warp_N * sub_warp_id;
                T*       bj = &index<T>(B, ldb, 0, j, false, ArrangeB);

                T rmem[M*M + 2*M*sub_warp_N];
                warp_impl_left_nonunit(A, lda, bj, ldb, M, sub_warp_N, Transpose, Fill, ArrangeA, ArrangeB, lane_id, sub_warp_NT, rmem);
            } else {
                unsigned warp_id = thread_id / 32;
                unsigned lane_id = thread_id % 32;
                // Just use a single warp
                if (warp_id == 0) {
                    // Compute the largest power of two for which N_per_warp is divisible, then use that many subwarps (has to be <= 32, however)
                    constexpr unsigned pow2_divisor_N = pow2_divisor(N);
                    constexpr unsigned num_sub_warp = (pow2_divisor_N > 32u) ? 32 : pow2_divisor_N;
                    constexpr unsigned sub_warp_NT = 32 / num_sub_warp; 
                    constexpr unsigned sub_warp_N   = N / num_sub_warp;

                    unsigned j  = sub_warp_N * (lane_id / sub_warp_NT);
                    T*       bj = &index<T>(B, ldb, 0, j, false, ArrangeB);

                    T rmem[M*M + 2*M*sub_warp_N];
                    warp_impl_left_nonunit(A, lda, bj, ldb, M, sub_warp_N, Transpose, Fill, ArrangeA, ArrangeB, lane_id % sub_warp_NT, sub_warp_NT, rmem);
                }
            }

        } else {
            // Large number of RHSs or tiny matrix
            if constexpr (N % NT == 0) {
                // Can evenly divide RHSs to threads
                // This has less smem IO than the next branch if N > NT
                constexpr unsigned thread_N = N / NT;
                unsigned           j        = thread_N * thread_id;

                T* bj = &index<T>(B, ldb, 0, j, false, ArrangeB);

                T rmem[M*M + 2*M*thread_N];
                thread_impl_left_nonunit(A, lda, bj, ldb, M, thread_N, Transpose, Fill, ArrangeA, ArrangeB, rmem);
            } else {
                // Just distribute RHSs round robin
                for (int j = thread_id; j < int(N); j += NT) {
                    T* bj = &index<T>(B, ldb, 0, j, false, ArrangeB);

                    T rmem[M*M + M*1];
                    thread_impl_left_nonunit(A, lda, bj, ldb, M, 1, Transpose, Fill, ArrangeA, ArrangeB, rmem);
                }
            }
        }
    }

    template<class T, unsigned M, unsigned N, side Side, diag Diag, transpose Transpose, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB>
    inline __device__ void thread_execute(const T* A, const unsigned lda, T* B, const unsigned ldb) {
        constexpr bool is_left   = (Side == side::left);
        constexpr bool unit_diag = (Diag == diag::unit);

        static_assert(is_left && !unit_diag, "TRMM only supports key cases");
        static_assert((Fill == fill_mode::lower && Transpose == transpose::non_transposed) ||
                      (Fill == fill_mode::upper && Transpose != transpose::non_transposed), "TRMM only supports key cases");

        T rmem[M*M + M*N];
        thread_impl_left_nonunit(A, lda, B, ldb, M, N, Transpose, Fill, ArrangeA, ArrangeB, rmem);
    }

} // namespace cusolverdx::detail::trmm

#endif // CUSOLVERDX_DATABASE_TRMM_CUH
