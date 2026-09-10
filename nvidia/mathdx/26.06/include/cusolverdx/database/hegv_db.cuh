// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HEGV_DB_CUH
#define CUSOLVERDX_DATABASE_HEGV_DB_CUH

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail::hegv {

    // Size thresholds for suggesting batches per block (initially aligned with hegst/heev)
    constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, bool compute_vectors, [[maybe_unused]] int Arch) {
        // tuned for H100
        if (T == type_enum::real_f32) {
            if (compute_vectors) {
                return {3, 6, 14, 14, 30};
            } else {
                return {3, 6, 20, 20, 40};
            }
        } else if (T == type_enum::real_f64) {
            if (compute_vectors) {
                return {3, 6, 14, 14, 28};
            } else {
                return {3, 6, 18, 18, 36};
            }
        } else if (T == type_enum::complex_f32) {
            if (compute_vectors) {
                return {3, 5, 14, 14, 28};
            } else {
                return {3, 5, 12, 12, 28};
            }
        } else { // complex_f64
            if (compute_vectors) {
                return {3, 5, 14, 14, 22};
            } else {
                return {3, 5, 10, 10, 26};
            }
        }
    }

    constexpr inline __device__ __host__ threshold_array<5> suggested_block_dim_size_thresholds(type_enum T, bool compute_vectors, [[maybe_unused]] int Arch) {
        // tuned for H100
        constexpr unsigned INF = unsigned(-1);
        if (T == type_enum::real_f32) {
            if (compute_vectors) {
                return {40, 40, 68, 80, 112};
            } else {
                return {40, 40, 64, 80, 112};
            }
        } else if (T == type_enum::real_f64) {
            if (compute_vectors) {
                return {40, 40, 68, 80, INF};
            } else {
                return {36, 40, 68, 80, INF};
            }
        } else if (T == type_enum::complex_f32) {
            if (compute_vectors) {
                return {28, 36, 60, 64, 80};
            } else {
                return {28, 40, 48, 56, 80};
            }
        } else { // complex_f64
            if (compute_vectors) {
                return {32, 40, 56, INF, INF};
            } else {
                return {32, 40, 56, 64, INF};
            }
        }
    }

} // namespace cusolverdx::detail::hegv

#endif // CUSOLVERDX_DATABASE_HEGV_DB_CUH
