// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM75_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM75_HPP

#include <cute/config.hpp>

#include "cublasdx/detail/system_checks.hpp"

#include "commondx/traits/numeric_traits.hpp"

namespace cublasdx {
    namespace detail {
        using cute::Int;
        using cute::Layout;
        using cute::Shape;
        using cute::Stride;
        using cute::Tensor;
        // SM75 MMAs

        // F16 = F16 * F16 + F16
        struct SM75_16x8x8_F16F16F16F16_TN {
            using DRegisters = uint32_t[2];
            using ARegisters = uint32_t[2];
            using BRegisters = uint32_t[1];
            using CRegisters = uint32_t[2];

            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& b0,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUTE_ARCH_MMA_SM75_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                             "{%0, %1},"
                             "{%2, %3},"
                             "{%4},"
                             "{%5, %6};\n"
                             : "=r"(d0), "=r"(d1)
                             : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1));
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM75_16x8x8_F16F16F16F16_TN without CUTE_ARCH_MMA_SM75_ENABLED");
#endif
            }
        };

        // F32 = F16 * F16 + F32
        struct SM75_16x8x8_F32F16F16F32_TN {
            using DRegisters = float[4];
            using ARegisters = uint32_t[2];
            using BRegisters = uint32_t[1];
            using CRegisters = float[4];

            CUTE_HOST_DEVICE static void fma(float&          d0,
                                             float&          d1,
                                             float&          d2,
                                             float&          d3,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& b0,
                                             float const&    c0,
                                             float const&    c1,
                                             float const&    c2,
                                             float const&    c3) {
#if defined(CUTE_ARCH_MMA_SM75_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                             "{%0,  %1,  %2,  %3},"
                             "{%4,  %5},"
                             "{%6},"
                             "{%7,  %8,  %9,  %10};\n"
                             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                             : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM75_16x8x8_F32F16F16F32_TN without CUTE_ARCH_MMA_SM75_ENABLED");
#endif
            }
        };
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM75_HPP
