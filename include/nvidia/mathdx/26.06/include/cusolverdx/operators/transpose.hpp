// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_TRANSPOSE_HPP
#define CUSOLVERDX_OPERATORS_TRANSPOSE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx {

    template<transpose Value>
    struct TransposeMode: commondx::detail::constant_operator_expression<transpose, Value> {};

    namespace detail {
        using default_transpose_operator = TransposeMode<non_trans>;
    } // namespace detail
}

namespace commondx::detail {
    template<cusolverdx::transpose Value>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::transpose, cusolverdx::TransposeMode<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::transpose Value>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::TransposeMode<Value>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::transpose;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_TRANSPOSE_HPP
