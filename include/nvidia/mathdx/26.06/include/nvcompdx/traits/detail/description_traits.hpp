// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <type_traits>

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/description_traits.hpp"

#include "nvcompdx/operators/operator_type.hpp"

namespace nvcompdx::detail {
// Extract operator from Description type
// Void if absent
template <operator_type OperatorType, class Description>
using get_t = commondx::detail::get_t<operator_type, OperatorType, Description>;

// Extract operator from Description type
// Default if absent
template <operator_type OperatorType, class Description, class Default = void>
using get_or_default_t = commondx::detail::get_or_default_t<operator_type, OperatorType, Description, Default>;

// Check if Description type contains operator
template <operator_type OperatorType, class Description>
using has_operator = commondx::detail::has_operator<operator_type, OperatorType, Description>;

template <operator_type OperatorType, class Description>
inline constexpr bool has_operator_v = has_operator<OperatorType, Description>::value;

// Check if Description type contains at most 1 operator
template <operator_type OperatorType, class Description>
using has_at_most_one_of = commondx::detail::has_at_most_one_of<operator_type, OperatorType, Description>;

template <operator_type OperatorType, class Description>
inline constexpr bool has_at_most_one_of_v = has_at_most_one_of<OperatorType, Description>::value;
} // namespace nvcompdx::detail
