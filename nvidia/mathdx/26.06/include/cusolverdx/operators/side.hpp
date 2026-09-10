// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_SIDE_HPP
#define CUSOLVERDX_OPERATORS_SIDE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx {

    template<side s>
    struct Side: commondx::detail::operator_expression {
        static_assert(s == side::left || s == side::right, "Side has to be left or right");

        static constexpr side value = s;
    };

    namespace detail {
        using default_side_operator = Side<side::left>;
    } // namespace detail
    
} // namespace cusolverdx

namespace commondx::detail {
    template<cusolverdx::side side>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::side, cusolverdx::Side<side>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::side side>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Side<side>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::side;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_SIDE_HPP
