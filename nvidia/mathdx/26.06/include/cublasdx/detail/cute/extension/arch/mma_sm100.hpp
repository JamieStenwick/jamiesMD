// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM100_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM100_HPP

#include <cute/arch/config.hpp>
#include <cute/arch/mma.hpp>

#include <cute/arch/simd_sm100.hpp>

namespace cublasdx {
    namespace detail {

        struct SM100_2x1x1_F32F32F32F32 {
            using DRegisters = float2[1];
            using ARegisters = float2[1];
            using BRegisters = float[1];
            using CRegisters = float2[1];

            CUTE_HOST_DEVICE static void fma(float2& d01, float2 const& a01, float const& b0, float2 const& c01) {
#if defined(CUTE_ARCH_FFMA2_SM100_ENABLED)
                cute::fma(d01, a01, make_float2(b0, b0), c01);
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM100_2x1x1_F32F32F32F32 without CUTE_ARCH_FFMA2_SM100_ENABLED");
#endif
            }
        };

        struct SM100_1x2x1_F32F32F32F32 {
            using DRegisters = float2[1];
            using ARegisters = float[1];
            using BRegisters = float2[1];
            using CRegisters = float2[1];

            CUTE_HOST_DEVICE static void fma(float2& d01, float const& a0, float2 const& b01, float2 const& c01) {
#if defined(CUTE_ARCH_FFMA2_SM100_ENABLED)
                cute::fma(d01, make_float2(a0, a0), b01, c01);
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM100_1x2x1_F32F32F32F32 without CUTE_ARCH_FFMA2_SM100_ENABLED");
#endif
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM100_HPP
