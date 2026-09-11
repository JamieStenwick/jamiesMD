// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "commondx/operators/block_dim.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"

#include "nvcompdx/config.hpp"
#include "nvcompdx/types.hpp"

namespace nvcompdx {
// Import BlockDim operator from commonDx
template <unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
struct BlockDim : public commondx::BlockDim<X, Y, Z> {
  static_assert(Y != 1 || Z == 1, "Threads in BlockDim must be contiguous, BlockDim<X, 1, Z> is incorrect");
};
} // namespace nvcompdx

namespace commondx::detail {
template <unsigned int X, unsigned int Y, unsigned int Z>
struct is_operator<nvcompdx::operator_type, nvcompdx::operator_type::block_dim, nvcompdx::BlockDim<X, Y, Z>>
    : nvcompdx::detail::std::true_type {
};
} // namespace commondx::detail
