// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HEEV_DB_CUH
#define CUSOLVERDX_DATABASE_HEEV_DB_CUH

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail::heev {

    // Size thresholds for suggesting batches per block
    constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, bool compute_vectors, [[maybe_unused]] int Arch) {
        if (T == type_enum::real_f32) {
            if (compute_vectors) {
                return {4, 15, 15, 18, 18};
            } else {
                return {4, 16, 22, 36, 44};
            }
        } else if (T == type_enum::real_f64) {
            if (compute_vectors) {
                return {4, 15, 15, 17, 20};
            } else {
                return {4, 12, 22, 32, 44};
            }
        } else if (T == type_enum::complex_f32) { //complex<float>
            if (compute_vectors) {
                return {4, 14, 15, 16, 36};
            } else {
                return {4, 10, 14, 22, 36};
            }
        } else { // complex<double>
            if (compute_vectors) {
                return {4, 11, 15, 15, 16};
            } else {
                return {4, 10, 14, 20, 28};
            }
        }
    }

    // Size thresholds for suggesting the block dim for a single batch
    constexpr inline __device__ __host__ threshold_array<5> suggested_block_dim_size_thresholds(type_enum T, bool compute_vectors, [[maybe_unused]] int Arch) {
        constexpr unsigned INF = unsigned(-1);
        // Experimentally tuned for H100-PCIe
        if (T == type_enum::real_f32) {
            // tuned for N=1:128:4
            if (compute_vectors) {
                return {48, 64, 76, INF, INF};
            } else {
                return {56, 64, 80, INF, INF};
            }
        } else if (T == type_enum::real_f64) {
            // tuned for N=1:128:4
            if (compute_vectors) {
                return {44, 64, 104, 108, INF};
            } else {
                return {56, 64, 116, 128, INF};
            }
        } else if (T == type_enum::complex_f32) { //complex<float>
            if (compute_vectors) {
                // tuned for N=1:128:4
                return {32, 32, 56, 104, 112};
            } else {
                // tuned for N=1:96:4
                return {44, 52, 56, 80, INF};
            }
        } else { // complex<double>
            // tuned for N=1:96:4
            if (compute_vectors) {
                return {32, 40, 80, 80, INF};
            } else {
                return {36, 52, 80, INF, INF};
            }
        }
    }

} // namespace cusolverdx::detail::heev

#endif // CUSOLVERDX_DATABASE_HEEV_DB_CUH

