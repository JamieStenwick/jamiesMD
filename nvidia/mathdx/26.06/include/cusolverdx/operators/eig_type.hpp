// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_EIG_TYPE_HPP
#define CUSOLVERDX_OPERATORS_EIG_TYPE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cusolverdx {

    template<int Value>
    struct EigType : public commondx::detail::constant_operator_expression<int, Value> {
        static_assert(Value == 1 || Value == 2 || Value == 3,
                      "EigType must be 1, 2, or 3, corresponding to the 3 modes for the Hermitian-definite generalized eigenvalue problem.");
    };

    namespace detail {
        using default_eig_type_operator = EigType<1>;
    } // namespace detail
} // namespace cusolverdx

// Register operators
namespace commondx::detail {
    template<int Value>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::eig_type, cusolverdx::EigType<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<int Value>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::EigType<Value>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::eig_type;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_EIG_TYPE_HPP
