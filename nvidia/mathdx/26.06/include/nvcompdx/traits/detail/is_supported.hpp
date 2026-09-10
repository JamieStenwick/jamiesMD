// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <type_traits>

#include "nvcompdx/types.hpp"
#include "nvcompdx/operators.hpp"
#include "nvcompdx/traits/detail/description_traits.hpp"
#include "nvcompdx/traits/detail/num_warps.hpp"
#include "nvcompdx/detail/comp_checks.hpp"

namespace nvcompdx {
namespace detail {
template <typename NVCOMPDX, unsigned int Architecture>
constexpr bool is_supported_impl() {
  // Traits
  constexpr datatype this_datatype = detail::get_t<operator_type::datatype, NVCOMPDX>::value;
  constexpr algorithm this_algorithm = detail::get_t<operator_type::algorithm, NVCOMPDX>::value;
  constexpr direction this_direction = detail::get_t<operator_type::direction, NVCOMPDX>::value;
  constexpr grouptype this_grouptype = detail::has_operator_v<operator_type::warp, NVCOMPDX> ? grouptype::warp
                                                                                             : grouptype::block;
  constexpr unsigned int this_num_warps = detail::num_warps_of_v<NVCOMPDX>;

  // Note:
  // Every other check (is_supported_datatype<>, is_supported_max_uncomp_chunk_size<>)
  // were already done while the description `NVCOMPDX` was created.
  return is_supported_shared_size<
    this_datatype,
    this_algorithm,
    this_direction,
    this_grouptype,
    this_num_warps,
    Architecture>::value;
}
} // namespace detail

// Check if a description is supported on a given CUDA architecture
template <class Description, unsigned int Architecture>
struct is_supported : detail::std::bool_constant<detail::is_supported_impl<Description, Architecture>()> {
};

template <class Description, unsigned int Architecture>
inline constexpr bool is_supported_v = is_supported<Description, Architecture>::value;
} // namespace nvcompdx
