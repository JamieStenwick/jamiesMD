// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_TYPE_HPP
#define CUSOLVERDX_OPERATORS_TYPE_HPP

#include "commondx/operators/type.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"

namespace cusolverdx {
    using type = commondx::data_type;

    template<type Value>
    struct Type: public commondx::DataType<Value> {};

    namespace detail {
        using default_type_operator = Type<type::real>;
    }
} // namespace cusolverdx


namespace commondx::detail {
    template<cusolverdx::type Value>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::type, cusolverdx::Type<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::type Value>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Type<Value>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::type;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_TYPE_HPP
