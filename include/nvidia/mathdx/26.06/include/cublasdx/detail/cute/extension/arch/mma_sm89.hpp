// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM89_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM89_HPP

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

        struct SM89_16x8x32_F32F8F8F32_TN {
            using DRegisters = float[4];
            using ARegisters = uint32_t[4];
            using BRegisters = uint32_t[2];
            using CRegisters = float[4];

            // fma() defined in derived class
        };

        struct SM89_16x8x16_F32F8F8F32_TN {
            using DRegisters = float[4];
            using ARegisters = uint32_t[2];
            using BRegisters = uint32_t[1];
            using CRegisters = float[4];

            // fma() defined in derived class
        };

        // SM89 F32 = F8 * F8 + F32 MMA

        // F32 = fe4m3 * fe4m3 + F32

        struct SM89_16x8x32_F32E4M3E4M3F32_TN: public SM89_16x8x32_F32F8F8F32_TN {
            CUTE_HOST_DEVICE static void fma(float&          d0,
                                             float&          d1,
                                             float&          d2,
                                             float&          d3,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& a2,
                                             uint32_t const& a3,
                                             uint32_t const& b0,
                                             uint32_t const& b1,
                                             float const&    c0,
                                             float const&    c1,
                                             float const&    c2,
                                             float const&    c3) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) // float
                             : "r"(a0),
                               "r"(a1),
                               "r"(a2),
                               "r"(a3), // uint32_t
                               "r"(b0),
                               "r"(b1), // uint32_t
                               "f"(c0),
                               "f"(c1),
                               "f"(c2),
                               "f"(c3) // float
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without SM89 ENABLED and NVCC 12.4 or above");
#endif
            }
        };

        // F32 = fe4m3 * fe5m2 + F32

        struct SM89_16x8x32_F32E4M3E5M2F32_TN: public SM89_16x8x32_F32F8F8F32_TN {
            CUTE_HOST_DEVICE static void fma(float&          d0,
                                             float&          d1,
                                             float&          d2,
                                             float&          d3,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& a2,
                                             uint32_t const& a3,
                                             uint32_t const& b0,
                                             uint32_t const& b1,
                                             float const&    c0,
                                             float const&    c1,
                                             float const&    c2,
                                             float const&    c3) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x32_F32E4M3E5M2F32_TN without SM89 ENABLED and NVCC 12.4 or above");
#endif
            }
        };

        // F32 = fe5m2 * fe4m3 + F32

        struct SM89_16x8x32_F32E5M2E4M3F32_TN: public SM89_16x8x32_F32F8F8F32_TN {
            CUTE_HOST_DEVICE static void fma(float&          d0,
                                             float&          d1,
                                             float&          d2,
                                             float&          d3,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& a2,
                                             uint32_t const& a3,
                                             uint32_t const& b0,
                                             uint32_t const& b1,
                                             float const&    c0,
                                             float const&    c1,
                                             float const&    c2,
                                             float const&    c3) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x32_F32E5M2E4M3F32_TN without SM89 ENABLED and NVCC 12.4 or above");
#endif
            }
        };

        // FP32 = fe5m2 * fe5m2 + F32
        struct SM89_16x8x32_F32E5M2E5M2F32_TN: public SM89_16x8x32_F32F8F8F32_TN {
            CUTE_HOST_DEVICE static void fma(float&          d0,
                                             float&          d1,
                                             float&          d2,
                                             float&          d3,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& a2,
                                             uint32_t const& a3,
                                             uint32_t const& b0,
                                             uint32_t const& b1,
                                             float const&    c0,
                                             float const&    c1,
                                             float const&    c2,
                                             float const&    c3) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x32_F32E5M2E5M2F32_TN without SM89 ENABLED and NVCC 12.4 or above");
#endif
            }
        };

        // 16x8x16 MMAs

        struct SM89_16x8x16_F32E4M3E4M3F32_TN: public SM89_16x8x16_F32F8F8F32_TN {
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
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
                             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) // float
                             : "r"(a0),
                               "r"(a1), // uint32_t
                               "r"(b0), // uint32_t
                               "f"(c0),
                               "f"(c1),
                               "f"(c2),
                               "f"(c3) // float
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x16_F32E4M3E4M3F32_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };

        // F32 = fe4m3 * fe5m2 + F32

        struct SM89_16x8x16_F32E4M3E5M2F32_TN: public SM89_16x8x16_F32F8F8F32_TN {
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
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e5m2.f32 "
                             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                             : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x16_F32E4M3E5M2F32_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };

        // F32 = fe5m2 * fe4m3 + F32

        struct SM89_16x8x16_F32E5M2E4M3F32_TN: public SM89_16x8x16_F32F8F8F32_TN {
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
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e4m3.f32 "
                             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                             : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x16_F32E5M2E4M3F32_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };

        // FP32 = fe5m2 * fe5m2 + F32
        struct SM89_16x8x16_F32E5M2E5M2F32_TN: public SM89_16x8x16_F32F8F8F32_TN {
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
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32 "
                             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                             : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x16_F32E5M2E5M2F32_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };


        struct SM89_16x8x32_F16F8F8F16_TN {
            using DRegisters = uint32_t[2];
            using ARegisters = uint32_t[4];
            using BRegisters = uint32_t[2];
            using CRegisters = uint32_t[2];

            // fma() defined in derived class
        };

        struct SM89_16x8x16_F16F8F8F16_TN {
            using DRegisters = uint32_t[2];
            using ARegisters = uint32_t[2];
            using BRegisters = uint32_t[1];
            using CRegisters = uint32_t[2];

            // fma() defined in derived class
        };
        // 16x8x32 MMAs
        // SM89 F16 = F8 * F8 + F16 MMA

        // F16 = fe4m3 * fe4m3 + F16

        struct SM89_16x8x32_F16E4M3E4M3F16_TN: public SM89_16x8x32_F16F8F8F16_TN {
            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& a2,
                                             uint32_t const& a3,
                                             uint32_t const& b0,
                                             uint32_t const& b1,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                             "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                             : "=r"(d0), "=r"(d1) // uint32_t
                             : "r"(a0),
                               "r"(a1),
                               "r"(a2),
                               "r"(a3), // uint32_t
                               "r"(b0),
                               "r"(b1), // uint32_t
                               "r"(c0),
                               "r"(c1) // uint32_t
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x32_F16E4M3E4M3F16_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };


        struct SM89_16x8x32_F16E4M3E5M2F16_TN: public SM89_16x8x32_F16F8F8F16_TN {
            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& a2,
                                             uint32_t const& a3,
                                             uint32_t const& b0,
                                             uint32_t const& b1,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16 "
                             "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                             : "=r"(d0), "=r"(d1) // uint32_t
                             : "r"(a0),
                               "r"(a1),
                               "r"(a2),
                               "r"(a3), // uint32_t
                               "r"(b0),
                               "r"(b1), // uint32_t
                               "r"(c0),
                               "r"(c1) // uint32_t
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x32_F16E4M3E5M2F16_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };


        struct SM89_16x8x32_F16E5M2E4M3F16_TN: public SM89_16x8x32_F16F8F8F16_TN {
            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& a2,
                                             uint32_t const& a3,
                                             uint32_t const& b0,
                                             uint32_t const& b1,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16 "
                             "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                             : "=r"(d0), "=r"(d1) // uint32_t
                             : "r"(a0),
                               "r"(a1),
                               "r"(a2),
                               "r"(a3), // uint32_t
                               "r"(b0),
                               "r"(b1), // uint32_t
                               "r"(c0),
                               "r"(c1) // uint32_t
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x32_F16E5M2E4M3F16_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };


        struct SM89_16x8x32_F16E5M2E5M2F16_TN: public SM89_16x8x32_F16F8F8F16_TN {
            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& a2,
                                             uint32_t const& a3,
                                             uint32_t const& b0,
                                             uint32_t const& b1,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16 "
                             "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                             : "=r"(d0), "=r"(d1) // uint32_t
                             : "r"(a0),
                               "r"(a1),
                               "r"(a2),
                               "r"(a3), // uint32_t
                               "r"(b0),
                               "r"(b1), // uint32_t
                               "r"(c0),
                               "r"(c1) // uint32_t
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x32_F16E5M2E5M2F16_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };

        // 16x8x16 MMAs
        // F16 = fe4m3 * fe4m3 + F16

        struct SM89_16x8x16_F16E4M3E4M3F16_TN: public SM89_16x8x16_F16F8F8F16_TN {
            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& b0,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16 "
                             "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
                             : "=r"(d0), "=r"(d1) // uint32_t
                             : "r"(a0),
                               "r"(a1), // uint32_t
                               "r"(b0), // uint32_t
                               "r"(c0),
                               "r"(c1) // uint32_t
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x16_F16E4M3E4M3F16_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };


        struct SM89_16x8x16_F16E4M3E5M2F16_TN: public SM89_16x8x16_F16F8F8F16_TN {
            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& b0,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e5m2.f16 "
                             "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
                             : "=r"(d0), "=r"(d1) // uint32_t
                             : "r"(a0),
                               "r"(a1), // uint32_t
                               "r"(b0), // uint32_t
                               "r"(c0),
                               "r"(c1) // uint32_t
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x16_F16E4M3E5M2F16_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };


        struct SM89_16x8x16_F16E5M2E4M3F16_TN: public SM89_16x8x16_F16F8F8F16_TN {
            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& b0,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e4m3.f16 "
                             "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
                             : "=r"(d0), "=r"(d1) // uint32_t
                             : "r"(a0),
                               "r"(a1), // uint32_t
                               "r"(b0), // uint32_t
                               "r"(c0),
                               "r"(c1) // uint32_t
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x16_F16E5M2E4M3F16_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };


        struct SM89_16x8x16_F16E5M2E5M2F16_TN: public SM89_16x8x16_F16F8F8F16_TN {
            CUTE_HOST_DEVICE static void fma(uint32_t&       d0,
                                             uint32_t&       d1,
                                             uint32_t const& a0,
                                             uint32_t const& a1,
                                             uint32_t const& b0,
                                             uint32_t const& c0,
                                             uint32_t const& c1) {
#if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16 "
                             "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
                             : "=r"(d0), "=r"(d1) // uint32_t
                             : "r"(a0),
                               "r"(a1), // uint32_t
                               "r"(b0), // uint32_t
                               "r"(c0),
                               "r"(c1) // uint32_t
                );
#else
                CUTE_INVALID_CONTROL_PATH(
                    "Attempting to use SM89_16x8x16_F16E5M2E5M2F16_TN without SM89 ENABLED and NVCC 12.8 or above");
#endif
            }
        };
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_SM89_HPP
