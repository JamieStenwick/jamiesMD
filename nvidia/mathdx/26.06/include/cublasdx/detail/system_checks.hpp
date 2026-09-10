// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_SYSTEM_CHECKS_HPP
#define CUBLASDX_DETAIL_SYSTEM_CHECKS_HPP

// We require target architecture to be Turing+ (only checking on device)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#    error "cuBLASDx requires GPU architecture sm_75 or higher"
#endif

#define AFFECTED_BY_NVBUG_5668269 (__CUDACC_VER_MAJOR__ == 13 && (__CUDACC_VER_MINOR__ == 0))
#define AFFECTED_BY_NVBUG_5430354 (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ <= 1)
#define AFFECTED_BY_NVBUG_5697031 (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ == 0)
#define AFFECTED_BY_NVBUG_6283425 (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)
#define AFFECTED_BY_NVBUG_6283118 (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#    define CUBLASDX_ARCH_MMA_SM89_ENABLED
#endif

#ifdef __CUDACC_RTC__

// NVRTC version check
#    ifndef CUBLASDX_IGNORE_DEPRECATED_COMPILER
#        if !(__CUDACC_VER_MAJOR__ >= 13)
#            error cuBLASDx requires NVRTC from CUDA Toolkit 13.0 or newer
#        endif
#    endif // CUBLASDX_IGNORE_DEPRECATED_COMPILER

// NVRTC compilation checks
#    ifndef CUBLASDX_IGNORE_DEPRECATED_COMPILER
static_assert((__CUDACC_VER_MAJOR__ >= 13), "cuBLASDx requires CUDA Runtime 13.0 or newer to work with NVRTC");
#    endif // CUBLASDX_IGNORE_DEPRECATED_COMPILER

#else
#    include <cuda.h>

// NVCC compilation

static_assert(CUDART_VERSION >= 13000, "cuBLASDx requires CUDA Runtime 13.0 or newer");
static_assert(CUDA_VERSION >= 13000, "cuBLASDx requires CUDA Toolkit 13.0 or newer");
#    ifdef __NVCC__
static_assert((__CUDACC_VER_MAJOR__ >= 13), "cuBLASDx requires NVCC 13.0 or newer");
#    endif

#    ifndef CUBLASDX_IGNORE_DEPRECATED_COMPILER

// Test for GCC 7+
#        if defined(__GNUC__) && !defined(__clang__)
#            if (__GNUC__ < 7)
#                error cuBLASDx requires GCC in version 7 or newer
#            endif
#        endif // __GNUC__

// Test for clang 9+
#        ifdef __clang__
#            if (__clang_major__ < 9)
#                error cuBLASDx requires clang in version 9 or newer (experimental support for clang as host compiler)
#            endif
#        endif // __clang__

// MSVC (Visual Studio) is not supported
#        ifdef _MSC_VER
#            error cuBLASDx does not support compilation with MSVC yet
#        endif // _MSC_VER

#    endif // CUBLASDX_IGNORE_DEPRECATED_COMPILER

#endif // __CUDACC_RTC__

// C++ Version
#ifndef CUBLASDX_IGNORE_DEPRECATED_DIALECT
#    if (__cplusplus < 201703L)
#        error cuBLASDx requires C++17 (or newer) enabled
#    endif
#endif // CUBLASDX_IGNORE_DEPRECATED_DIALECT

#endif // CUBLASDX_DETAIL_SYSTEM_CHECKS_HPP
