// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// ordering of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_OPERATORS_ORDERING_HPP
#define CURANDDX_OPERATORS_ORDERING_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace curanddx {

    enum class order
    {
        strict, 
        curand_legacy, 
        subsequence,
    };
    inline constexpr auto strict  = order::strict;
    inline constexpr auto curand_legacy   = order::curand_legacy;
    inline constexpr auto subsequence = order::subsequence;

    template<order Value>
    struct Ordering: public commondx::detail::constant_operator_expression<order, Value> {};

    namespace detail {
        using default_curanddx_ordering_operator = Ordering<order::subsequence>;
    } // namespace detail
} // namespace curanddx

namespace commondx::detail {
    template<curanddx::order Value>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::ordering, curanddx::Ordering<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<curanddx::order Value>
    struct get_operator_type<curanddx::operator_type, curanddx::Ordering<Value>> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::ordering;
    };
} // namespace commondx::detail


#endif // CURANDDX_OPERATORS_ORDERING_HPP
