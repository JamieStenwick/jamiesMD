// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_HPP
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_HPP

#if !defined(__CUDACC_RTC__)

#include "cublasdx/detail/pipeline/emulation/exponents.hpp"
#include "cublasdx/detail/pipeline/emulation/slicing.hpp"
#include "cublasdx/detail/pipeline/emulation/emulation_tile_pipeline.hpp"
#include "cublasdx/detail/pipeline/emulation/emulation_device_pipeline.hpp"
#include "cublasdx/detail/pipeline/emulation/suggest_pipeline.hpp"

#endif // !defined(__CUDACC_RTC__)

#endif // CUBLASDX_DETAIL_PIPELINE_EMULATION_HPP
