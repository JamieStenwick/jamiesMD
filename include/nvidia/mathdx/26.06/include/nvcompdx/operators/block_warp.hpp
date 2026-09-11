// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"

#include "nvcompdx/config.hpp"
#include "nvcompdx/types.hpp"

namespace nvcompdx {
template <unsigned int NumWarps, bool Complete>
struct BlockWarp : commondx::detail::operator_expression {
  static_assert(NumWarps >= 1, "Number of warps in the thread block must be greater than or equal to 1");
  static constexpr unsigned int num_warps = NumWarps;
  static constexpr bool complete = Complete;
};
} // namespace nvcompdx

namespace commondx::detail {
template <unsigned int NumWarps, bool Complete>
struct is_operator<nvcompdx::operator_type, nvcompdx::operator_type::block_warp, nvcompdx::BlockWarp<NumWarps, Complete>>
    : nvcompdx::detail::std::true_type {
};
} // namespace commondx::detail
