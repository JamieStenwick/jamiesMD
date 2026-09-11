// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_COMMONDX_CONFIG_HPP
#define CUSOLVERDX_DETAIL_COMMONDX_CONFIG_HPP

// STL namespace alias
#include "commondx/detail/config.hpp"

#if defined(__CUDA_ARCH__)
#define CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(CUSOLVERDX_TYPE)                                                                 \
  if constexpr (cusolverdx::sm_of_v<CUSOLVERDX_TYPE> != __CUDA_ARCH__) {                                                   \
    return;                                                                                                            \
  }
#else
#define CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(CUSOLVERDX_TYPE)
#endif // __CUDA_ARCH__

#endif // CUSOLVERDX_DETAIL_COMMONDX_CONFIG_HPP
