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

namespace nvcompdx::detail {
// is_complete_description
namespace is_complete_description_impl {
template <class Description, class Enable = void>
struct helper : detail::std::false_type {
};

template <template <class...> class Description, class... Types>
struct helper<
  Description<Types...>,
  detail::std::enable_if_t<commondx::detail::is_description_expression<Description<Types...>>::value>> {
  using description_type = Description<Types...>;

  // Defaults (only for compilation)
  using default_direction = Direction<direction::compress>;

  // Description operators
  // SM
  static constexpr bool has_this_sm = has_operator_v<operator_type::sm, description_type>;
  // Algorithm
  static constexpr bool has_this_algorithm = has_operator_v<operator_type::algorithm, description_type>;
  // Direction
  static constexpr bool has_this_direction = has_operator_v<operator_type::direction, description_type>;
  // Data type
  static constexpr bool has_this_datatype = has_operator_v<operator_type::datatype, description_type>;
  // Maximum uncompressed chunk size (mandatory for compression)
  static constexpr bool max_uncomp_chunk_size_needed =
    has_this_direction &&
    detail::get_or_default_t<operator_type::direction, description_type, default_direction>::value ==
      direction::compress;
  static constexpr bool has_max_uncomp_chunk_size_if_needed =
    (max_uncomp_chunk_size_needed && has_operator_v<operator_type::max_uncomp_chunk_size, description_type>) ||
    !max_uncomp_chunk_size_needed;

  static constexpr bool value = has_this_sm && has_this_algorithm && has_this_direction && has_this_datatype &&
                                has_max_uncomp_chunk_size_if_needed;
};
} // namespace is_complete_description_impl

template <class Description>
struct is_complete_description
    : detail::std::integral_constant<bool, is_complete_description_impl::helper<Description>::value> {
};

// is_complete_execution
namespace is_complete_execution_impl {
template <class Description, class Enable = void>
struct helper : detail::std::false_type {
};

template <template <class...> class Description, class... Types>
struct helper<
  Description<Types...>,
  detail::std::enable_if_t<commondx::detail::is_description_expression<Description<Types...>>::value>> {
  using description_type = Description<Types...>;

  // Execution operators
  // Warp
  static constexpr bool has_this_warp = has_operator_v<operator_type::warp, description_type>;
  // Block
  static constexpr bool has_this_block = has_operator_v<operator_type::block, description_type>;
  // Block description (mandatory for block mode)
  static constexpr bool block_description_needed = has_this_block;
  static constexpr bool has_this_block_description_if_needed =
    (block_description_needed && (has_operator_v<operator_type::block_dim, description_type> ||
                                  has_operator_v<operator_type::block_warp, description_type>)) ||
    !block_description_needed;

  static constexpr bool value = (has_this_warp || has_this_block) && has_this_block_description_if_needed;
};
} // namespace is_complete_execution_impl

template <class Description>
struct is_complete_execution
    : detail::std::integral_constant<bool, is_complete_execution_impl::helper<Description>::value> {
};

// is_complete_execution_description
template <class Description>
struct is_complete_execution_description
    : detail::std::integral_constant<
        bool,
        is_complete_description_impl::helper<Description>::value &&
          is_complete_execution_impl::helper<Description>::value> {
};
} // namespace nvcompdx::detail
