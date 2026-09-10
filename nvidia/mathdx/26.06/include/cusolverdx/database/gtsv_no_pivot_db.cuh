// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.


#ifndef CUSOLVERDX_DATABASE_GTSV_NO_PIVOT_DB_CUH
#define CUSOLVERDX_DATABASE_GTSV_NO_PIVOT_DB_CUH

// header with the arch-specific suggestions and thresholds

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail::gtsv_no_pivot {

    constexpr inline __device__ __host__ unsigned tiny_threshold(type_enum T, [[maybe_unused]] int Arch) {
        if (T == type_enum::real_f32) {
            return 12;
        } else if (T == type_enum::real_f64) {
            return 8;
        } else if (T == type_enum::complex_f32) { //complex<float>
            return 8;
        } else { // complex<double>
            return 8;
        }
    }

    // Size thresholds for suggesting the block dim for a single batch
    constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
        // Experimentally tuned for H100-PCIe
        if (T == type_enum::real_f32) {
            return {4, 28, 40, 100, 208};
        } else if (T == type_enum::real_f64) {
            return {4, 16, 28, 40, 112};
        } else if (T == type_enum::complex_f32) { //complex<float>
            return {4, 16, 36, 72, 238};
        } else { // complex<double>
            return {3, 16, 20, 56, 112};
        }
    }

} // namespace cusolverdx::detail::gtsv_no_pivot

#endif // CUSOLVERDX_DATABASE_GTSV_NO_PIVOT_DB_CUH
