// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_BDSVD_DB_CUH
#define CUSOLVERDX_DATABASE_BDSVD_DB_CUH

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail::bdsvd {

    // Size thresholds for suggesting batches per block
    constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, bool compute_vectors, [[maybe_unused]] int Arch) {
        // Taken from htev without further tuning
        if (!compute_vectors) {
            return {64, 64, 64, 64, 64};
        } else {
            if (T == type_enum::real_f32) {
                return {8, 15, 15, 15, 15};
            } else if (T == type_enum::real_f64) {
                return {4, 15, 15, 15, 15};
            } else {
                // Need C++23 to use if consteval
                // if consteval {
                //     static_assert(type_enum_is_real(T), "BDSVD does not support complex types");
                // }
                return {4, 15, 15, 15, 15};
            }
        }
    }


} // namespace cusolverdx::detail::bdsvd

#endif // CUSOLVERDX_DATABASE_BDSVD_DB_CUH

