// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HETRD_DB_CUH
#define CUSOLVERDX_DATABASE_HETRD_DB_CUH

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail::hetrd {

    ////////// thresholds for implementation dispatch ////////////

    // Threshold at which point the thread impl is always optimal
    constexpr inline __device__ __host__ unsigned tiny_threshold(type_enum T, [[maybe_unused]] int Arch) {
        // Based on H100-PCIe
        if (T == type_enum::real_f32) {
            return 4;
        } else if (T == type_enum::real_f64) {
            return 4;
        } else if (T == type_enum::complex_f32) { //complex<float>
            return 5;
        } else { // complex<double>
            return 8;
        }
    }

    // Threshold to enable "big" optimization
    constexpr inline __device__ __host__ unsigned big_threshold(type_enum T, [[maybe_unused]] int Arch) {
        constexpr unsigned INF = unsigned(-1);
        // Based on H100-PCIe
        if (T == type_enum::real_f32) {
            return 116;
        } else if (T == type_enum::real_f64) {
            return 84;
        } else if (T == type_enum::complex_f32) { //complex<float>
            return INF;
        } else { // complex<double>
            return 40;
        }
    }

} // namespace cusolverdx::detail::hetrd

#endif // CUSOLVERDX_DATABASE_HETRD_DB_CUH

