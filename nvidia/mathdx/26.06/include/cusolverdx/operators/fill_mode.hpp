// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_FILL_MODE_HPP
#define CUSOLVERDX_OPERATORS_FILL_MODE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx {

    template<fill_mode Value>
    struct FillMode: public commondx::detail::constant_operator_expression<fill_mode, Value> {};

    namespace detail {
        using default_fill_mode_operator = FillMode<fill_mode::lower>;
    } // namespace detail
} // namespace cusolverdx

namespace commondx::detail {
    template<cusolverdx::fill_mode Value>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::fill_mode, cusolverdx::FillMode<Value>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::fill_mode Value>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::FillMode<Value>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::fill_mode;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_FILL_MODE_HPP
