// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <type_traits>
#include <tuple>

#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/dx_traits.hpp"

#include "nvcompdx/operators.hpp"
#include "nvcompdx/types.hpp"
#include "nvcompdx/detail/comp_description_fd.hpp"
#include "nvcompdx/traits/detail/is_complete.hpp"
#include "nvcompdx/traits/detail/is_supported.hpp"
#include "nvcompdx/traits/detail/num_warps.hpp"

namespace nvcompdx {
// ------------------
// Execution checkers
// ------------------

// is_warp
template <class Description>
struct is_warp {
public:
  static constexpr bool value = detail::has_operator_v<operator_type::warp, Description>;
};

template <class Description>
inline constexpr bool is_warp_v = is_warp<Description>::value;

// is_block
template <class Description>
struct is_block {
public:
  static constexpr bool value = detail::has_operator_v<operator_type::block, Description>;
};

template <class Description>
inline constexpr bool is_block_v = is_block<Description>::value;

// ----------------
// Operator getters
// ----------------

// algorithm_of
template <class Description>
struct algorithm_of {
private:
  static constexpr bool has_algorithm = detail::has_operator_v<operator_type::algorithm, Description>;
  static_assert(has_algorithm, "Description does not have Algorithm defined");

public:
  using value_type = algorithm;
  static constexpr value_type value = detail::get_t<operator_type::algorithm, Description>::value;
};

template <class Description>
inline constexpr algorithm algorithm_of_v = algorithm_of<Description>::value;

// block_dim_of
template <class Description>
using block_dim_of = commondx::block_dim_of<operator_type, Description>;

template <class Description>
inline constexpr dim3 block_dim_of_v = block_dim_of<Description>::value;

// block_warp_of
template <class Description>
struct block_warp_of {
private:
  static constexpr bool has_block_warp = detail::has_operator_v<operator_type::block_warp, Description>;
  static_assert(has_block_warp, "Description does not have BlockWarp defined");

public:
  using value_type = detail::std::tuple<unsigned int, bool>;
  static constexpr unsigned int num_warps = detail::get_t<operator_type::block_warp, Description>::num_warps;
  static constexpr bool complete = detail::get_t<operator_type::block_warp, Description>::complete;
  static constexpr value_type value = value_type{num_warps, complete};
  constexpr operator value_type() const noexcept { return value; }
};

template <class Description>
inline constexpr detail::std::tuple<unsigned int, bool> block_warp_of_v = block_warp_of<Description>::value;

// datatype_of
template <class Description>
struct datatype_of {
private:
  static constexpr bool has_datatype = detail::has_operator_v<operator_type::datatype, Description>;
  static_assert(has_datatype, "Description does not have Datatype defined");

public:
  using value_type = datatype;
  static constexpr value_type value = detail::get_t<operator_type::datatype, Description>::value;
};

template <class Description>
inline constexpr datatype datatype_of_v = datatype_of<Description>::value;

// direction_of
template <class Description>
struct direction_of {
private:
  static constexpr bool has_direction = detail::has_operator_v<operator_type::direction, Description>;
  static_assert(has_direction, "Description does not have Direction defined");

public:
  using value_type = direction;
  static constexpr value_type value = detail::get_t<operator_type::direction, Description>::value;
};

template <class Description>
inline constexpr direction direction_of_v = direction_of<Description>::value;

// max_uncomp_chunk_size_of
template <class Description>
struct max_uncomp_chunk_size_of {
private:
  static constexpr bool has_max_uncomp_chunk_size =
    detail::has_operator_v<operator_type::max_uncomp_chunk_size, Description>;
  static_assert(has_max_uncomp_chunk_size, "Description does not have MaxUncompChunkSize defined");

public:
  using value_type = unsigned long long;
  static constexpr value_type value = detail::get_t<operator_type::max_uncomp_chunk_size, Description>::value;
};

template <class Description>
inline constexpr unsigned long long max_uncomp_chunk_size_of_v = max_uncomp_chunk_size_of<Description>::value;

// sm_of
template <class Description>
using sm_of = commondx::sm_of<operator_type, Description>;

template <class Description>
inline constexpr unsigned int sm_of_v = sm_of<Description>::value;

// --------------------------
// General Description traits
// --------------------------

template <class Description>
using is_comp = commondx::is_dx_expression<Description>;

template <class Description>
inline constexpr bool is_comp_v = is_comp<Description>::value;

template <class Description>
using is_comp_execution = commondx::is_complete_dx_expression<Description, detail::is_complete_execution>;

template <class Description>
inline constexpr bool is_comp_execution_v = is_comp_execution<Description>::value;

template <class Description>
using is_complete_comp = commondx::is_complete_dx_expression<Description, detail::is_complete_description>;

template <class Description>
inline constexpr bool is_complete_comp_v = is_complete_comp<Description>::value;

template <class Description>
using is_complete_comp_execution =
  commondx::is_complete_dx_expression<Description, detail::is_complete_execution_description>;

template <class Description>
inline constexpr bool is_complete_comp_execution_v = is_complete_comp_execution<Description>::value;

template <class Description>
using extract_comp_description = commondx::extract_dx_description<detail::comp_description, Description, operator_type>;

template <class Description>
using extract_comp_description_t = typename extract_comp_description<Description>::type;

} // namespace nvcompdx
