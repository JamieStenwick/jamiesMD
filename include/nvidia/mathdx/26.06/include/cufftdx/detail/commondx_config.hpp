// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_COMMONDX_CONFIG_HPP
#define CUFFTDX_DETAIL_COMMONDX_CONFIG_HPP

// STL namespace alias
#include "commondx/detail/config.hpp"

#if defined(__CUDA_ARCH__)
#    define CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM(CUFFTDX_TYPE)                  \
        if constexpr (cufftdx::sm_of_v<CUFFTDX_TYPE> != __CUDA_ARCH__) {    \
            return;                                                         \
        }
#else
#    define CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM(CUFFTDX_TYPE)
#endif // __CUDA_ARCH__

#endif // CUFFTDX_DETAIL_COMMONDX_CONFIG_HPP
