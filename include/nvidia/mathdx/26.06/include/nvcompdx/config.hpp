// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "commondx/detail/config.hpp"

#if defined(__CUDA_ARCH__)
#define NVCOMPDX_SKIP_IF_NOT_APPLICABLE_SM(NVCOMPDX_TYPE)                                                              \
  if constexpr (nvcompdx::sm_of_v<NVCOMPDX_TYPE> != __CUDA_ARCH__) {                                                   \
    return;                                                                                                            \
  }
#else
#define NVCOMPDX_SKIP_IF_NOT_APPLICABLE_SM(NVCOMPDX_TYPE)
#endif // __CUDA_ARCH__

#ifdef __CUDACC_RTC__
namespace cuda::std {}
#endif // __CUDACC_RTC__

namespace nvcompdx::detail {

#ifdef __CUDACC_RTC__
namespace std = ::cuda::std;
#else
namespace std = ::std;
#endif // __CUDACC_RTC__

} // namespace nvcompdx::detail
