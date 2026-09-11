// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GETRF_PARTIAL_PIVOT_DB_CUH
#define CUSOLVERDX_DATABASE_GETRF_PARTIAL_PIVOT_DB_CUH

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

// Tuning factors are currently just borrowed from Cholesky
#include "cusolverdx/database/potrf_db.cuh"

namespace cusolverdx {
    namespace detail {
        namespace getrf_partial_pivot {

            // Size thresholds for suggesting batches per block
            constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
                // Take tuning factors from lu_np until lu_pp can be tuned
                // Slightly modified based on a few tests
                if (T == type_enum::real_f32) {
                    return {5*5, 6*6, 7*7, 15*15, 24*24};
                } else if (T == type_enum::real_f64) {
                    return {5*5, 6*6, 9*9, 15*15, 24*24};
                } else if (T == type_enum::complex_f32) { //complex<float>
                    return {5*5, 13*13, 16*16, 20*20, 24*24};
                } else { // complex<double>
                    return {5*5, 9*9, 14*14, 18*18, 21*21};
                }
            }

            // Size thresholds for suggesting the block dim for a single batch
            constexpr inline __device__ __host__ threshold_array<3> suggested_block_dim_size_thresholds(type_enum T, int Arch) {
                if (Arch >= 1000) {
                    if (T == type_enum::real_f32) {
                        // tuned for 1 <= N <= 128
                        return {40*40, 64*64, 116*116};
                    } else if (T == type_enum::real_f64) {
                        // tuned for 1 <= N <= 128
                        return {44*44, 63*63, 128*128};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        // tuned for 1 <= N <= 96
                        return {16*16, 56*56, 80*80};
                    } else { // complex<double>
                        // tuned for 1 <= N <= 96
                        return {16*16, 40*40, 56*56};
                    }
                } else {
                    // Take tuning factors from lu_np until lu_pp can be tuned
                    if (T == type_enum::real_f32) {
                        // tuned for 1 <= N <= 128
                        return {53*53, 72*72, 95*95};
                    } else if (T == type_enum::real_f64) {
                        // tuned for 1 <= N <= 128
                        return {48*48, 63*63, 88*88};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        // tuned for 1 <= N <= 96
                        return {16*16, 56*56, 80*80};
                    } else { // complex<double>
                        // tuned for 1 <= N <= 96
                        return {16*16, 40*40, 56*56};
                    }
                }
            }

            ////////// thresholds for implementation dispatch ////////////

            constexpr inline __device__ __host__ unsigned tiny_threshold(type_enum T, int Arch) {
                // Take tuning factors from Cholesky until lu_pp can be tuned
                auto chol_tol = potrf::tiny_threshold(T, Arch);
                return chol_tol * chol_tol;
            }

            // Used to determine when the problem is too large for the partial-warp implementation
            constexpr inline __device__ __host__ unsigned small_threshold(type_enum T, int Arch) {
                // Take tuning factors from Cholesky until lu_pp can be tuned
                auto chol_tol = potrf::small_threshold(T, Arch);
                return chol_tol * chol_tol;
            }

            constexpr inline __device__ __host__ unsigned med_threshold(type_enum T, int Arch) {
                // Take tuning factors from Cholesky until lu_pp can be tuned
                auto chol_tol = potrf::med_threshold(T, Arch);
                return chol_tol * chol_tol;
            }

        } // namespace getrf_partial_pivot
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_GETRF_PARTIAL_PIVOT_DB_CUH
