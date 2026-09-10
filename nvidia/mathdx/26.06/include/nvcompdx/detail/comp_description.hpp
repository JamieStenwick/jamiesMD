// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <type_traits>

#include "commondx/traits/detail/get.hpp"
#include "commondx/detail/expressions.hpp"

#include "nvcompdx/operators.hpp"
#include "nvcompdx/traits.hpp"
#include "nvcompdx/detail/utils.hpp"

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

namespace nvcompdx::detail {

template <class... Operators>
class comp_operator_wrapper : public commondx::detail::description_expression
{
};

template <class... Operators>
class comp_description : public commondx::detail::description_expression
{
  using description_type = comp_operator_wrapper<Operators...>;

protected:
  /// ---- Traits
  /// ---- Description Traits

  // Algorithm
  // * Default value: NONE
  // * Dummy value: algorithm::lz4
  static constexpr bool has_algorithm = has_operator<operator_type::algorithm, description_type>::value;
  using dummy_default_algorithm = Algorithm<algorithm::lz4>;
  using this_algorithm = get_or_default_t<operator_type::algorithm, description_type, dummy_default_algorithm>;
  static constexpr auto this_algorithm_v = this_algorithm::value;

  // Direction
  // * Default value: NONE
  // * Dummy value: direction::compress
  static constexpr bool has_direction = has_operator<operator_type::direction, description_type>::value;
  using dummy_default_direction = Direction<direction::compress>;
  using this_direction = get_or_default_t<operator_type::direction, description_type, dummy_default_direction>;
  static constexpr auto this_direction_v = this_direction::value;
  // Helper boolean for SFINAE (MSVC compatibility - cast to int to avoid operator== lookup issues)
  static constexpr bool is_compress_direction =
    (static_cast<int>(this_direction_v) == static_cast<int>(direction::compress));

  // Data type
  // * Default value: NONE
  // * Dummy value: datatype::uint8
  static constexpr bool has_datatype = has_operator<operator_type::datatype, description_type>::value;
  using dummy_default_datatype = DataType<datatype::uint8>;
  using this_datatype = get_or_default_t<operator_type::datatype, description_type, dummy_default_datatype>;
  static constexpr auto this_datatype_v = this_datatype::value;

  // Maximum uncompressed chunk size
  // * Default value: NONE
  // * Dummy value: 16k
  static constexpr bool has_max_uncomp_chunk_size =
    has_operator<operator_type::max_uncomp_chunk_size, description_type>::value;
  using dummy_default_max_uncomp_chunk_size = MaxUncompChunkSize<1 << 14>;
  using this_max_uncomp_chunk_size =
    get_or_default_t<operator_type::max_uncomp_chunk_size, description_type, dummy_default_max_uncomp_chunk_size>;
  static constexpr auto this_max_uncomp_chunk_size_v = this_max_uncomp_chunk_size::value;

  // SM
  // * Default value: NONE
  // * Dummy value: 750
  static constexpr bool has_sm = has_operator<operator_type::sm, description_type>::value;
  using dummy_default_sm = SM<750>;
  using this_sm = get_or_default_t<operator_type::sm, description_type, dummy_default_sm>;
  static constexpr auto this_sm_v = this_sm::value;

  // Is it a complete description?
  static constexpr bool is_complete_v = is_complete_description<description_type>::value;

  /// ---- Execution Traits
  // Warp
  static constexpr bool has_warp = has_operator_v<operator_type::warp, description_type>;

  // Block
  static constexpr bool has_block = has_operator_v<operator_type::block, description_type>;

  // BlockDim
  // * Default value: NONE
  // * Dummy value: 32 x 1 x 1
  static constexpr bool has_block_dim = has_operator<operator_type::block_dim, description_type>::value;
  using dummy_default_block_dim = BlockDim<32, 1, 1>;
  static constexpr auto this_block_dim_x =
    get_or_default_t<operator_type::block_dim, description_type, dummy_default_block_dim>::value.x;
  static constexpr auto this_block_dim_y =
    get_or_default_t<operator_type::block_dim, description_type, dummy_default_block_dim>::value.y;
  static constexpr auto this_block_dim_z =
    get_or_default_t<operator_type::block_dim, description_type, dummy_default_block_dim>::value.z;

  using this_block_dim = BlockDim<this_block_dim_x, this_block_dim_y, this_block_dim_z>;
  static constexpr auto this_block_dim_v = this_block_dim::value;

  // BlockWarp
  // * Default value: NONE
  // * Dummy value (1, true)
  static constexpr bool has_block_warp = has_operator<operator_type::block_warp, description_type>::value;
  using dummy_default_block_warp = BlockWarp<1, true>;
  static constexpr auto this_block_warp_num_warps =
    get_or_default_t<operator_type::block_warp, description_type, dummy_default_block_warp>::num_warps;
  static constexpr auto this_block_warp_complete =
    get_or_default_t<operator_type::block_warp, description_type, dummy_default_block_warp>::complete;

  using this_block_warp = BlockWarp<this_block_warp_num_warps, this_block_warp_complete>;

  /// ---- Aggregate Traits

  // Group type
  static constexpr auto this_grouptype_v = has_warp ? grouptype::warp : grouptype::block;

  // Number of warps participating
  static constexpr auto this_aggregate_num_warps_v = detail::min(
    16u,
    has_block_dim ? (this_block_dim::flat_size / 32) : (has_block_warp ? this_block_warp_num_warps : 1u));

  // Is it a complete thread block?
  static constexpr auto this_aggregate_complete_v =
    has_block_dim ? (this_block_dim::flat_size % 32 == 0 && this_block_dim::flat_size / 32 <= 16u)
                  : (has_block_warp ? (this_block_warp_complete && this_block_warp_num_warps <= 16u) : false);

  /// ---- Constraints

  // We can only have one of each option
  static constexpr bool has_one_block_dim = has_at_most_one_of_v<operator_type::block_dim, description_type>;
  static constexpr bool has_one_block_warp = has_at_most_one_of_v<operator_type::block_warp, description_type>;
  static constexpr bool has_one_warp = has_at_most_one_of_v<operator_type::warp, description_type>;
  static constexpr bool has_one_block = has_at_most_one_of_v<operator_type::block, description_type>;
  static constexpr bool has_one_comp_algorithm = has_at_most_one_of_v<operator_type::algorithm, description_type>;
  static constexpr bool has_one_comp_datatype = has_at_most_one_of_v<operator_type::datatype, description_type>;
  static constexpr bool has_one_comp_direction = has_at_most_one_of_v<operator_type::direction, description_type>;
  static constexpr bool has_one_comp_max_uncomp_chunk_size =
    has_at_most_one_of_v<operator_type::max_uncomp_chunk_size, description_type>;
  static constexpr bool has_one_sm = has_at_most_one_of_v<operator_type::sm, description_type>;

  static_assert(has_one_block_dim, "Can't create nvcompdx function with two BlockDim<> expressions");
  static_assert(has_one_block_warp, "Can't create nvcompdx function with two BlockWarp<> expressions");
  static_assert(has_one_warp, "Can't create nvcompdx function with two Warp<> expressions");
  static_assert(has_one_block, "Can't create nvcompdx function with two Block<> expressions");
  static_assert(has_one_comp_algorithm, "Can't create nvcompdx function with two Algorithm<> expressions");
  static_assert(has_one_comp_datatype, "Can't create nvcompdx function with two DataType<> expressions");
  static_assert(has_one_comp_direction, "Can't create nvcompdx function with two Direction<> expressions");
  static_assert(
    has_one_comp_max_uncomp_chunk_size,
    "Can't create nvcompdx function with two MaxUncompChunkSize<> expressions");
  static_assert(has_one_sm, "Can't create nvcompdx function with two SM<> expressions");

  /// ---- Additional static checks

  // Check if at least 1 warp is participating in the problem
  static constexpr bool valid_block_dim = this_aggregate_num_warps_v >= 1;
  static_assert(
    valid_block_dim,
    "The provided block dimension is invalid. The block must have at least 1 warp, or 32 threads.");

  // Check if the datatype is supported by the algorithm
  static constexpr bool valid_datatype = is_supported_datatype<this_datatype_v, this_algorithm_v>::value;
  static_assert(
    !has_datatype || !has_algorithm || valid_datatype,
    "The provided data type is not supported by the selected algorithm.");

  // Check if the specified max uncomp. chunk size is supported
  static constexpr bool valid_max_uncomp_chunk_size =
    is_supported_max_uncomp_chunk_size<this_datatype_v, this_algorithm_v, this_max_uncomp_chunk_size_v>::value;
  static_assert(
    !has_max_uncomp_chunk_size || valid_max_uncomp_chunk_size,
    "The provided maximum uncompressed chunk size falls outside the supported range.");

  // Check if the problem's scratch space fits in shared memory
  static constexpr bool valid_shared_size = is_supported_shared_size<
    this_datatype_v,
    this_algorithm_v,
    this_direction_v,
    this_grouptype_v,
    this_aggregate_num_warps_v,
    this_sm_v>::value;
  static_assert(
    !has_sm || valid_shared_size,
    "The provided specification makes this problem not fit into shared memory "
    "available on the specified architecture.");

  /// ---- End of Constraints
public:
  static constexpr __device__ __host__ bool is_complete() { return is_complete_v; }
};

template <>
class comp_description<> : public commondx::detail::description_expression
{
};
} // namespace nvcompdx::detail

#undef STRINGIFY
#undef XSTRINGIFY
