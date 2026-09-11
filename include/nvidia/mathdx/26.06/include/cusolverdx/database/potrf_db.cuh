// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_POTRF_DB_CUH
#define CUSOLVERDX_DATABASE_POTRF_DB_CUH

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx {
    namespace detail {
        namespace potrf {

            // Size thresholds for suggesting batches per block
            constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, int Arch) {
                if (Arch >= 900) {
                    // Experimentally tuned for H100-PCIe
                    if (T == type_enum::real_f32) {
                        return {5, 6, 7, 15, 16};
                    } else if (T == type_enum::real_f64) {
                        return {4, 6, 7, 12, 14};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        return {3, 5, 6, 12, 16};
                    } else { // complex<double>
                        return {3, 5, 6, 12, 14};
                    }
                } else {
                    // Experimentally tuned for A100-PCIe
                    if (T == type_enum::real_f32) {
                        return {3, 5, 7, 10, 12};
                    } else if (T == type_enum::real_f64) {
                        return {3, 4, 7, 10, 14};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        return {3, 4, 7, 10, 14};
                    } else { // complex<double>
                        return {2, 3, 6, 8, 14};
                    }
                }
            }

            // Size thresholds for suggesting the block dim for a single batch
            constexpr inline __device__ __host__ threshold_array<5> suggested_block_dim_size_thresholds(type_enum T, int Arch) {
                constexpr unsigned INF = unsigned(-1);
                if (Arch >= 1000) {
                    // Experimentally tuned for B200
                    if (T == type_enum::real_f32) {
                        // tuned for N=1:128:4
                        return {40, 72, 96, 116, 124};
                    } else if (T == type_enum::real_f64) {
                        // tuned for N=1:128:4
                        return {48, 72, 84, 108, 128};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        // tuned for N=1:96:4
                        return {24, 56, 64, 96, INF};
                    } else { // complex<double>
                        // tuned for N=1:96:4
                        return {24, 56, 80, 88, 96};
                    }
                } else if (Arch >= 900) {
                    // Experimentally tuned for H100-PCIe
                    if (T == type_enum::real_f32) {
                        // tuned for 1 <= N <= 128
                        return {43, 64, 111, INF, INF};
                    } else if (T == type_enum::real_f64) {
                        // tuned for 1 <= N <= 96
                        return {25, 78, INF, INF, INF};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        // tuned for 1 <= N <= 96
                        return {25, 78, INF, INF, INF};
                    } else { // complex<double>
                        // tuned for 1 <= N <= 64
                        return {24, 48, INF, INF, INF};
                    }
                } else {
                    // Experimentally tuned for A100-PCIe for sizes <= 100
                    // Larger sizes use SM90 tuning
                    if (T == type_enum::real_f32) {
                        return {46, 72, 111, INF, INF};
                    } else if (T == type_enum::real_f64) {
                        return {22, 72, INF, INF, INF};
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        return {21, 57, INF, INF, INF};
                    } else { // complex<double>
                        return {25, 56, INF, INF, INF};
                    }
                }
            }

            ////////// thresholds for implementation dispatch ////////////

            constexpr inline __device__ __host__ unsigned tiny_threshold(type_enum T, int Arch) {
                if (Arch >= 900) {
                    // Based on H100-PCIe
                    return type_enum_is_real(T) ? 6 : 5;
                } else if (Arch == 860) {
                    // Based on RTX A6000
                    if (T == type_enum::real_f32) {
                        return 6;
                    } else if (T == type_enum::real_f64) {
                        return 4;
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        return 5;
                    } else { // complex<double>
                        return 2;
                    }
                } else {
                    // Based on A100-PCIe
                    return type_enum_is_real(T) ? 6 : 4;
                }
            }

            // Used to determine when the problem is too large for the partial-warp implementation
            constexpr inline __device__ __host__ unsigned small_threshold(type_enum T, int Arch) {
                if (Arch == 860) {
                    // Based on RTX A6000
                    if (T == type_enum::real_f32) {
                        return 60;
                    } else if (T == type_enum::real_f64) {
                        return 60;
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        return 40;
                    } else { // complex<double>
                        return 50;
                    }
                } else {
                    // Based on H100-PCIe and A100-PCIe
                    // Cross-over point was the same for both cards
                    if (T == type_enum::real_f32) {
                        return 54;
                    } else if (T == type_enum::real_f64) {
                        return 54;
                    } else if (T == type_enum::complex_f32) { //complex<float>
                        return 40;
                    } else { // complex<double>
                        return 36;
                    }
                }
            }

            constexpr inline __device__ __host__ unsigned med_threshold([[maybe_unused]] type_enum T, [[maybe_unused]] int Arch) {
                // Based on H100
                return 64;
            }

        } // namespace potrf
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_POTRF_DB_CUH
