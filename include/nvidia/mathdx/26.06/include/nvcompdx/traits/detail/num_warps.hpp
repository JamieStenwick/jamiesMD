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
#include "nvcompdx/traits/detail/description_traits.hpp"
#include "nvcompdx/detail/utils.hpp"

namespace nvcompdx::detail {
// num_warps
namespace num_warps_of_impl {
template <class Description, class Enable = void>
struct helper : detail::std::integral_constant<int, 0> {
};

template <template <class...> class Description, class... Types>
struct helper<
  Description<Types...>,
  detail::std::enable_if_t<commondx::detail::is_description_expression<Description<Types...>>::value>> {
  using description_type = Description<Types...>;

  // Defaults
  using default_block_warp = BlockWarp<1, true>;
  using default_block_dim = BlockDim<32, 1, 1>;

  // Execution
  static constexpr bool has_this_warp = has_operator_v<operator_type::warp, description_type>;
  static constexpr bool has_this_block = has_operator_v<operator_type::block, description_type>;

  // Block description
  static constexpr bool has_block_warp = has_operator_v<operator_type::block_warp, description_type>;
  static constexpr bool has_block_dim = has_operator_v<operator_type::block_dim, description_type>;
  static constexpr unsigned int block_warp_value =
    has_block_warp
      ? detail::get_or_default_t<operator_type::block_warp, description_type, default_block_warp>::num_warps
      : 0u;
  static constexpr unsigned int block_dim_value =
    has_block_dim
      ? detail::get_or_default_t<operator_type::block_dim, description_type, default_block_dim>::flat_size / 32u
      : 0u;

  static constexpr unsigned int value =
    detail::min(16u, has_this_warp ? 1 : (has_block_dim ? block_dim_value : block_warp_value));
};
} // namespace num_warps_of_impl

template <class Description>
struct num_warps_of : detail::std::integral_constant<int, num_warps_of_impl::helper<Description>::value> {
};

template <class Description>
inline constexpr unsigned int num_warps_of_v = num_warps_of<Description>::value;
} // namespace nvcompdx::detail
