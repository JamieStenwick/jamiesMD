// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HEGST_DB_CUH
#define CUSOLVERDX_DATABASE_HEGST_DB_CUH

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail::hegst {

    // Size thresholds for suggesting batches per block
    constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
        // Tuned on H100-PCIe
        if (T == type_enum::real_f32) {
            return {4, 5, 14, 14, 40};
        } else if (T == type_enum::real_f64) {
            return {2, 3, 9, 10, 28};
        } else if (T == type_enum::complex_f32) { //complex<float>
            return {2, 3, 10, 10, 28};
        } else { // complex<double>
            return {2, 2, 9, 9, 12};
        }
    }

    // Size thresholds for suggesting the block dim for a single batch
    constexpr inline __device__ __host__ threshold_array<5> suggested_block_dim_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
        // Tuned on H100-PCIe
        constexpr unsigned INF = unsigned(-1);
        if (T == type_enum::real_f32) {
            return {40, 48, 64, 80, 96};
        } else if (T == type_enum::real_f64) {
            return {28, 48, 64, 80, 96};
        } else if (T == type_enum::complex_f32) { //complex<float>
            return {28, 40, 48, 64, 80};
        } else { // complex<double>
            return {12, 32, 40, 64, INF};
        }
    }

    constexpr inline __device__ __host__ unsigned tiny_threshold([[maybe_unused]] type_enum T, [[maybe_unused]] int Arch) {
        return 4;
    }

} // namespace cusolverdx::detail::hegst

#endif // CUSOLVERDX_DATABASE_HEGST_DB_CUH


