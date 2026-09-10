// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

// We require target architecture to be Turing+ (only checking on device)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#error "nvCOMPDx requires GPU architecture sm_75 or higher"
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

#ifdef __CUDACC_RTC__

// NVRTC compilation checks
#ifndef NVCOMPDX_IGNORE_DEPRECATED_COMPILER
static_assert(__CUDACC_VER_MAJOR__ >= 13, "nvCOMPDx requires CUDA Runtime 13.0 or newer to work with NVRTC");
#endif // NVCOMPDX_IGNORE_DEPRECATED_COMPILER

#else

#include <cuda.h>

// NVCC compilation
static_assert(CUDART_VERSION >= 13000, "nvCOMPDx requires CUDA Runtime 13.0.0 or newer");
static_assert(CUDA_VERSION >= 13000, "nvCOMPDx requires CUDA Toolkit 13.0.0 or newer");
#ifdef __NVCC__
static_assert(__CUDACC_VER_MAJOR__ >= 13, "nvCOMPDx requires NVCC 13.0 or newer");
#endif // __NVCC__

#ifndef NVCOMPDX_IGNORE_DEPRECATED_COMPILER
// Test for GCC 7+
#if defined(__GNUC__) && !defined(__clang__)
#if (__GNUC__ < 7)
#error "nvCOMPDx requires GCC in version 7 or newer"
#endif
#endif // __GNUC__

// Test for clang 9+
#ifdef __clang__
#if (__clang_major__ < 9)
#error "nvCOMPDx requires clang in version 9 or newer (experimental support for clang as host compiler)"
#endif
#endif // __clang__

// Test for NVHPC
#ifdef __NVCOMPILER
#error "nvCOMPDx does not support nvc++ yet"
#endif // __NVCOMPILER

#endif // NVCOMPDX_IGNORE_DEPRECATED_COMPILER

#endif // __CUDACC_RTC__

// C++ Version
#ifndef NVCOMPDX_IGNORE_DEPRECATED_DIALECT
#if (__cplusplus < 201703L)
#error "nvCOMPDx requires C++17 (or newer) enabled"
#endif
#endif // NVCOMPDX_IGNORE_DEPRECATED_DIALECT
