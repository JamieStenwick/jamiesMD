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
template <algorithm Value>
struct Algorithm : public commondx::detail::constant_operator_expression<algorithm, Value> {
};
} // namespace nvcompdx

namespace commondx::detail {
template <nvcompdx::algorithm Value>
struct is_operator<nvcompdx::operator_type, nvcompdx::operator_type::algorithm, nvcompdx::Algorithm<Value>>
    : nvcompdx::detail::std::true_type {
};
} // namespace commondx::detail
