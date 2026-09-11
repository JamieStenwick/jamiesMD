// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GEQRF_DB_CUH
#define CUSOLVERDX_DATABASE_GEQRF_DB_CUH

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx {
    namespace detail {
        namespace geqrf {

            // Size thresholds for suggesting batches per block
            constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
                if (T == type_enum::real_f32) {
                    return {3*3, 8*8, 12*12, 22*22, 24*24};
                } else if (T == type_enum::real_f64) {
                    return {3*3, 6*6, 8*8, 22*22, 24*24};
                } else if (T == type_enum::complex_f32) {
                    return {3*3, 6*6, 9*9, 20*20, 24*24};
                } else { // complex_f64
                    return {3*3, 6*6, 14*14, 16*16, 24*24};
                }
            }

            // Size thresholds for suggesting the block dim for a single batch
            constexpr inline __device__ __host__ threshold_array<5> suggested_block_dim_size_thresholds(type_enum T, int Arch) {
                constexpr unsigned INF = unsigned(-1);
                if (Arch >= 1000) {
                    if (T == type_enum::real_f32) {
                        // tuned for 1 <= N <= 128
                        return {36*36, 64*64, 80*80, 96*96, 112*112};
                    } else if (T == type_enum::real_f64) {
                        // tuned for 1 <= N <= 128
                        return {36*36, 60*60, 80*80, 96*96, 112*112};
                    } else if (T == type_enum::complex_f32) {
                        // tuned for 1 <= N <= 128
                        return {32*32, 52*52, 64*64, 80*80, INF};
                    } else { // complex_f64
                        // tuned for 1 <= N <= 120
                        return {32*32, 52*52, 64*64, 80*80, INF};
                    }
                } else { // Tuned for H100
                    if (T == type_enum::real_f32) {
                        return {44*44, 56*56, 76*76, 112*112, INF};
                    } else if (T == type_enum::real_f64) {
                        return {40*40, 48*48, 56*56, 100*100, INF};
                    } else if (T == type_enum::complex_f32) {
                        return {28*28, 36*36, 56*56, 84*84, INF};
                    } else { // complex_f64
                        return {24*24, 32*32, 64*64, 80*80, INF};
                    }
                }
            }

        } // namespace geqrf
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_GEQRF_DB_CUH
