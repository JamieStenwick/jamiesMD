// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GEQRS_DB_CUH
#define CUSOLVERDX_DATABASE_GEQRS_DB_CUH

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail {

    namespace geqrs {


        // Size thresholds for suggesting batches per block
        constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
            if (T == type_enum::real_f32) {
                return {3 * 3, 7 * 7, 12 * 12, 22 * 22, 24 * 24};
            } else if (T == type_enum::real_f64) {
                return {3 * 3, 6 * 6, 8 * 8, 22 * 22, 24 * 24};
            } else if (T == type_enum::complex_f32) {
                return {3 * 3, 6 * 6, 9 * 9, 11 * 11, 15 * 15};
            } else { // complex_f64
                return {3 * 3, 6 * 6, 9 * 9, 11 * 11, 15 * 15};
            }
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
    } // namespace geqrs
} // namespace cusolverdx::detail

#endif // CUSOLVERDX_DATABASE_GEQRS_DB_CUH
