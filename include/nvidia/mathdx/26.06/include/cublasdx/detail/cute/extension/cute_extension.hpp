// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_CUTE_EXTENSION_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_CUTE_EXTENSION_HPP

// Custom MMA extensions
#include "cublasdx/detail/cute/extension/atom/mma_traits.hpp"
#include "cublasdx/detail/cute/extension/atom/mma_traits_sm75.hpp"
#include "cublasdx/detail/cute/extension/atom/mma_traits_sm89.hpp"
#include "cublasdx/detail/cute/extension/atom/mma_traits_sm100.hpp"

// Complex MMA must be kept at the end because it needs previously defined MMA traits
#include "cublasdx/detail/cute/extension/atom/complex_mma_traits.hpp"


// Custom copy extensions
#include "cublasdx/detail/cute/extension/algorithm/copy.hpp"
#include "cublasdx/detail/cute/extension/algorithm/cooperative_copy.hpp"

// Various functional and conversion utilities
#include "cublasdx/detail/cute/extension/cute_compatibility.hpp"

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_CUTE_EXTENSION_HPP
