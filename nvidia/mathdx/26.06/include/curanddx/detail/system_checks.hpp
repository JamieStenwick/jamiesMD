// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_SYSTEM_CHECKS_HPP
#define CURANDDX_DETAIL_SYSTEM_CHECKS_HPP

// We require target architecture to be Turing or newer (only checking on device)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    #error "cuRANDDx requires GPU architecture sm_75 or higher"
#endif

#ifdef __CUDACC_RTC__

    // NVRTC version check
    #ifndef CURANDDX_IGNORE_DEPRECATED_COMPILER
        #if !(__CUDACC_VER_MAJOR__ >= 13)
            #error "cuRANDDx requires NVRTC from CUDA Toolkit 13.0 or newer"
        #endif
    #endif // CURANDDX_IGNORE_DEPRECATED_COMPILER

    // NVRTC compilation checks
    #ifndef CURANDDX_IGNORE_DEPRECATED_COMPILER
static_assert((__CUDACC_VER_MAJOR__ >= 13), "cuRANDDx requires CUDA Runtime 13.0 or newer to work with NVRTC");
    #endif // CURANDDX_IGNORE_DEPRECATED_COMPILER

#else
    #include <cuda.h>

// NVCC compilation
static_assert(CUDART_VERSION >= 13000, "cuRANDDx requires CUDA Runtime 13.0 or newer");
static_assert(CUDA_VERSION >= 13000, "cuRANDDx requires CUDA Toolkit 13.0 or newer");
    #ifdef __NVCC__
static_assert(__CUDACC_VER_MAJOR__ >= 13, "cuRANDDx requires NVCC 13.0 or newer");
    #endif

    #ifndef CURANDDX_IGNORE_DEPRECATED_COMPILER

        // Test for GCC 7+
        #if defined(__GNUC__) && !defined(__clang__)
            #if (__GNUC__ < 7)
                #error "cuRANDDx requires GCC in version 7 or newer"
            #endif
        #endif // __GNUC__

        // Test for clang 9+
        #ifdef __clang__
            #if (__clang_major__ < 9)
                #error "cuRANDDx requires clang in version 9 or newer (experimental support for clang as host compiler)"
            #endif
        #endif // __clang__

        // MSVC (Visual Studio) is not supported
        #ifdef _MSC_VER
            #error "cuRANDDx does not support compilation with MSVC yet"
        #endif // _MSC_VER

    #endif // CURANDDX_IGNORE_DEPRECATED_COMPILER

#endif // __CUDACC_RTC__

// C++ Version
#ifndef CURANDDX_IGNORE_DEPRECATED_DIALECT
    #if (__cplusplus < 201703L)
        #error "cuRANDDx requires C++17 (or newer) enabled"
    #endif
#endif // CURANDDX_IGNORE_DEPRECATED_DIALECT

#endif // CURANDDX_DETAIL_SYSTEM_CHECKS_HPP
