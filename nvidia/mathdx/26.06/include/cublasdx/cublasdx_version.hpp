// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_VERSION_HPP
#define CUBLASDX_VERSION_HPP

/// \def CUBLASDX_VERSION
/// \brief CUBLASDX library version
///
///
/// @note
/// CUBLASDX_VERSION / 10000 - major version <br/>
/// CUBLASDX_VERSION / 100 % 100 - minor version <br/>
/// CUBLASDX_VERSION % 100 - patch level <br/>
#define CUBLASDX_VERSION 701

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define CUBLASDX_VERSION_MAJOR 0
#define CUBLASDX_VERSION_MINOR 7
#define CUBLASDX_VERSION_PATCH 1

#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \def CUBLASDX_CUTLASS_VERSION
/// \brief Version of the CUTLASS library that cuBLASDx was built against
///
/// Encoded as major * 10000 + minor * 100 + patch.
/// When using a custom CUTLASS, define this macro before including any
/// cuBLASDx header so that the library selects the correct code-path
/// for the CUTLASS API in use.
#ifndef CUBLASDX_CUTLASS_VERSION
#define CUBLASDX_CUTLASS_VERSION 40502
#endif

#endif // CUBLASDX_VERSION_HPP
