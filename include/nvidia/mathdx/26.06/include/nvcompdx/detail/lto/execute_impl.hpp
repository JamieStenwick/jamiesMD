// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

// Support for host compilers in host-only setups
#ifndef __host__
#define __host__
#endif // __host__
#ifndef __device__
#define __device__
#endif // __device__

// LZ4
#include "nvcompdx/detail/lto/lz4/compress_device.hpp"
#include "nvcompdx/detail/lto/lz4/decompress_device.hpp"
// ANS
#include "nvcompdx/detail/lto/ans/compress_device.hpp"
#include "nvcompdx/detail/lto/ans/decompress_device.hpp"
// Fallback
#ifndef NVCOMPDX_DETAIL_IGNORE_FALLBACK
#include "nvcompdx/detail/lto/common_fallback.hpp"
#endif // NVCOMPDX_DETAIL_IGNORE_FALLBACK
