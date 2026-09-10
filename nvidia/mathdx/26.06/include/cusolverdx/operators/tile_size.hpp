// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_TILE_SIZE_HPP
#define CUSOLVERDX_OPERATORS_TILE_SIZE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"

namespace cusolverdx {

    template<unsigned Value>
    struct TileSize : public commondx::detail::constant_operator_expression<unsigned, Value> {};

    namespace detail {
        using default_tile_size_operator = TileSize<64>;
    } // namespace detail
} // namespace cusolverdx

namespace commondx::detail {
    template<unsigned Value>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::tile_size, cusolverdx::TileSize<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned Value>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::TileSize<Value>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::tile_size;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_TILE_SIZE_HPP
