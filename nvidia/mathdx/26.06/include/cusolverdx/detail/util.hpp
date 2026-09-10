// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_UTIL_HPP
#define CUSOLVERDX_DETAIL_UTIL_HPP

namespace cusolverdx {
    namespace detail {

        // constexpr versions of min and max
        template<typename T1, typename T2>
        __host__ __device__ constexpr auto const_min(T1 a, T2 b) {
            return (a <= b) ? a : b;
        }
        template<typename T1, typename T2>
        __host__ __device__ constexpr auto const_max(T1 a, T2 b) {
            return (a >= b) ? a : b;
        }

    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_UTIL_HPP
