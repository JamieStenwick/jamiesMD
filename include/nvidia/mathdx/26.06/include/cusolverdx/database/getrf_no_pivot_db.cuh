// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GETRF_NO_PIVOT_DB_CUH
#define CUSOLVERDX_DATABASE_GETRF_NO_PIVOT_DB_CUH

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

// Tuning factors are currently borrowed in part from Cholesky
#include "cusolverdx/database/potrf_db.cuh"

namespace cusolverdx {
    namespace detail {
        namespace getrf_no_pivot {

            // Size thresholds for suggesting batches per block
            constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
                // Experimentally tuned for H100-PCIe
                if (T == type_enum::real_f32) {
                    return {5*5, 6*6, 7*7, 15*15, 32*32};
                } else if (T == type_enum::real_f64) {
                    return {5*5, 6*6, 9*9, 15*15, 32*32};
                } else if (T == type_enum::complex_f32) { //complex<float>
                    return {5*5, 13*13, 16*16, 20*20, 28*28};
                } else { // complex<double>
                    return {5*5, 9*9, 14*14, 18*18, 21*21};
                }
            }

            // Size thresholds for suggesting the block dim for a single batch
            constexpr inline __device__ __host__ threshold_array<4> suggested_block_dim_size_thresholds(type_enum T, int Arch) {
                constexpr unsigned INF = unsigned(-1);
                if (Arch >= 1000) {
                    // Scanned BlockDims for B200 computelab-next
                    if (T == type_enum::real_f32) {
                        // tuned for 1 <= N <= 128
                        return {53*53, 72*72, 95*95, 115*115};
                    } else if (T == type_enum::real_f64) {
                        // tuned for 1 <= N <= 128
                        return {48*48, 63*63, 99*99, 115*115};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        // tuned for 1 <= N <= 96
                        return {28*28, 59*59, 67*67, 80*80};
                    } else { // complex<double>
                        // tuned for 1 <= N <= 96
                        return {23*23, 43*43, 75*75, 80*80};
                    }
                } else {
                    // Experimentally tuned for H100-PCIe
                    if (T == type_enum::real_f32) {
                        // tuned for 1 <= N <= 128
                        return {53*53, 72*72, 95*95, INF};
                    } else if (T == type_enum::real_f64) {
                        // tuned for 1 <= N <= 128
                        return {48*48, 63*63, 88*88, INF};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        // tuned for 1 <= N <= 96
                        return {16*16, 56*56, 80*80, INF};
                    } else { // complex<double>
                        // tuned for 1 <= N <= 96
                        return {16*16, 40*40, 56*56, INF};
                    }
                }
            }

            ////////// thresholds for implementation dispatch ////////////

            constexpr inline __device__ __host__ unsigned tiny_threshold(type_enum T, int Arch) {
                // Take tuning factors from Cholesky until lu_np can be tuned
                auto chol_tol = potrf::tiny_threshold(T, Arch);
                return chol_tol * chol_tol;
            }

            // Used to determine when the problem is too large for the partial-warp implementation
            constexpr inline __device__ __host__ unsigned small_threshold(type_enum T, int Arch) {
                // Take tuning factors from Cholesky until lu_np can be tuned
                auto chol_tol = potrf::small_threshold(T, Arch);
                return chol_tol * chol_tol;
            }

            constexpr inline __device__ __host__ unsigned med_threshold(type_enum T, int Arch) {
                // Take tuning factors from Cholesky until lu_np can be tuned
                auto chol_tol = potrf::med_threshold(T, Arch);
                return chol_tol * chol_tol;
            }

        } // namespace getrf_no_pivot
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_GETRF_NO_PIVOT_DB_CUH
