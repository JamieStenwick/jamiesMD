// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GEQRS_CUH
#define CUSOLVERDX_DATABASE_GEQRS_CUH

// Dispatch logic for solve step of gels

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/indexing.cuh"

#include "cusolverdx/database/trsm.cuh"
#include "cusolverdx/database/unmqr.cuh"
#include "cusolverdx/database/unmlq.cuh"
#include "cusolverdx/database/geqrs_db.cuh"

namespace cusolverdx::detail {

    namespace geqrs {

        template<class T, unsigned m, unsigned n, unsigned k, arrangement arr_a, arrangement arr_b, transpose trans_a, unsigned NT, unsigned bpb>
        __device__ void block_execute(const T* A, unsigned lda, const T* tau, T* B, unsigned ldb, unsigned tid) {
            static_assert(m >= n, "QR solve is only valid for m>=n");

            constexpr auto trans = is_real_v<T> ? transpose::transposed : transpose::conj_transposed;

            // Because trsm only uses part of A & B, we need to use an explicit batch stride
            const unsigned strideA = (arr_a == arrangement::col_major) ? lda * n : m * lda;
            const unsigned strideB = (arr_b == arrangement::col_major) ? ldb * k : const_max(m, n) * ldb;

            if (trans_a == transpose::non_transposed) {
                // minimize ||b - Ax||_2
                // x = R^-1 Q^C b
                unmqr::block_execute<T, m, k, n, side::left, trans, arr_a, arr_b, NT, bpb>(A, lda, tau, B, ldb, tid);

                trsm::block_execute<T, n, k, side::left, diag::non_unit, transpose::non_transposed, fill_mode::upper, arr_a, arr_b, NT, bpb>(A, lda, strideA, B, ldb, strideB, tid);

            } else {
                // minimize ||x||_2 s.t., b = Ax
                // x = Q R^-C b

                trsm::block_execute<T, n, k, side::left, diag::non_unit, trans, fill_mode::upper, arr_a, arr_b, NT, bpb>(A, lda, strideA, B, ldb, strideB, tid);

                // TODO this could be unpacked to get better thread utilization
                for (int b = 0; b < int(bpb); ++b) {
                    for (int j = 0; j < int(k); ++j) {
                        for (int i = tid + n; i < int(m); i += NT) {
                            if constexpr (is_real_v<T>) {
                                index(B + b * strideB, ldb, i, j, arr_b) = 0.0;
                            } else {
                                index(B + b * strideB, ldb, i, j, arr_b) = {0.0, 0.0};
                            }
                        }
                    }
                }
                __syncthreads();

                unmqr::block_execute<T, m, k, n, side::left, transpose::non_transposed, arr_a, arr_b, NT, bpb>(A, lda, tau, B, ldb, tid);
            }
        }

        template<class T, unsigned m, unsigned n, unsigned k, arrangement arr_a, arrangement arr_b, transpose trans_a>
        __device__ void thread_execute(const T* A, unsigned lda, const T* tau, T* B, unsigned ldb) {
            static_assert(m >= n, "QR solve is only valid for m>=n");

            constexpr auto trans = is_real_v<T> ? transpose::transposed : transpose::conj_transposed;

            if (trans_a == transpose::non_transposed) {
                // minimize ||b - Ax||_2
                // x = R^-1 Q^C b
                unmqr::thread_execute<T, m, k, n, side::left, trans, arr_a, arr_b>(A, lda, tau, B, ldb);

                trsm::thread_execute<T, n, k, side::left, diag::non_unit, transpose::non_transposed, fill_mode::upper, arr_a, arr_b>(A, lda, B, ldb);

            } else {
                // minimize ||x||_2 s.t., b = Ax
                // x = Q R^-C b

                trsm::thread_execute<T, n, k, side::left, diag::non_unit, trans, fill_mode::upper, arr_a, arr_b>(A, lda, B, ldb);

                for (int j = 0; j < int(k); ++j) {
                    for (int i = n; i < int(m); i++) {
                        if constexpr (is_real_v<T>) {
                            index(B, ldb, i, j, arr_b) = 0.0;
                        } else {
                            index(B, ldb, i, j, arr_b) = {0.0, 0.0};
                        }
                    }
                }

                unmqr::thread_execute<T, m, k, n, side::left, transpose::non_transposed, arr_a, arr_b>(A, lda, tau, B, ldb);
            }
        }
    } // namespace geqrs

    namespace gelqs {
        template<class T, unsigned m, unsigned n, unsigned k, arrangement arr_a, arrangement arr_b, transpose trans_a, unsigned NT, unsigned bpb>
        __device__ void block_execute(const T* A, unsigned lda, const T* tau, T* B, unsigned ldb, unsigned tid) {
            static_assert(m <= n, "LQ solve is only valid for m<=n");

            constexpr auto trans = is_real_v<T> ? transpose::transposed : transpose::conj_transposed;

            // Because trsm only uses part of A & B, we need to use an explicit batch stride
            const unsigned strideA = (arr_a == arrangement::col_major) ? lda * n : m * lda;
            const unsigned strideB = (arr_b == arrangement::col_major) ? ldb * k : const_max(m, n) * ldb;

            if (trans_a == transpose::non_transposed) {
                // minimize ||x||_2 s.t., b = Ax
                // x = Q R^-C b

                trsm::block_execute<T, m, k, side::left, diag::non_unit, transpose::non_transposed, fill_mode::lower, arr_a, arr_b, NT, bpb>(A, lda, strideA, B, ldb, strideB, tid);

                // TODO this could be unpacked to get better thread utilization
                for (int b = 0; b < int(bpb); ++b) {
                    for (int j = 0; j < int(k); ++j) {
                        for (int i = tid + m; i < int(n); i += NT) {
                            if constexpr (is_real_v<T>) {
                                index(B + b * strideB, ldb, i, j, arr_b) = 0.0;
                            } else {
                                index(B + b * strideB, ldb, i, j, arr_b) = {0.0, 0.0};
                            }
                        }
                    }
                }
                __syncthreads();

                unmlq::block_execute<T, n, k, m, side::left, trans, arr_a, arr_b, NT, bpb>(A, lda, tau, B, ldb, tid);
            } else {
                // minimize ||b - Ax||_2
                // x = R^-1 Q^C b
                unmlq::block_execute<T, n, k, m, side::left, transpose::non_transposed, arr_a, arr_b, NT, bpb>(A, lda, tau, B, ldb, tid);

                trsm::block_execute<T, m, k, side::left, diag::non_unit, trans, fill_mode::lower, arr_a, arr_b, NT, bpb>(A, lda, strideA, B, ldb, strideB, tid);
            }
        }

        template<class T, unsigned m, unsigned n, unsigned k, arrangement arr_a, arrangement arr_b, transpose trans_a>
        __device__ void thread_execute(const T* A, unsigned lda, const T* tau, T* B, unsigned ldb) {
            static_assert(m <= n, "LQ solve is only valid for m<=n");

            constexpr auto trans = is_real_v<T> ? transpose::transposed : transpose::conj_transposed;

            if (trans_a == transpose::non_transposed) {
                // minimize ||x||_2 s.t., b = Ax
                // x = Q R^-C b

                trsm::thread_execute<T, m, k, side::left, diag::non_unit, transpose::non_transposed, fill_mode::lower, arr_a, arr_b>(A, lda, B, ldb);

                for (int j = 0; j < int(k); ++j) {
                    for (int i = m; i < int(n); i++) {
                        if constexpr (is_real_v<T>) {
                            index(B, ldb, i, j, arr_b) = 0.0;
                        } else {
                            index(B, ldb, i, j, arr_b) = {0.0, 0.0};
                        }
                    }
                }
                unmlq::thread_execute<T, n, k, m, side::left, trans, arr_a, arr_b>(A, lda, tau, B, ldb);
            } else {
                // minimize ||b - Ax||_2
                // x = R^-1 Q^C b
                unmlq::thread_execute<T, n, k, m, side::left, transpose::non_transposed, arr_a, arr_b>(A, lda, tau, B, ldb);

                trsm::thread_execute<T, m, k, side::left, diag::non_unit, trans, fill_mode::lower, arr_a, arr_b>(A, lda, B, ldb);
            }
        }
    } // namespace gelqs

} // namespace cusolverdx::detail
#endif // CUSOLVERDX_DATABASE_GEQRS_CUH
