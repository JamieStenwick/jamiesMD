// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// ordering of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_OPERATORS_PHILOX_ROUNDS_HPP
#define CURANDDX_OPERATORS_PHILOX_ROUNDS_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace curanddx {

    template<unsigned int Value>
    struct PhiloxRounds: public commondx::detail::constant_operator_expression<unsigned int, Value> {
#ifndef CURANDDX_PHILOX_ROUNDS_CHECK_DISABLED
        static_assert((Value > 5 && Value < 11), "Philox rounds must be an integer between [6, 10]");
#endif
    };

    namespace detail {
        using default_curanddx_philox_rounds_operator = PhiloxRounds<10>;
    }

} // namespace curanddx

namespace commondx::detail {
    template<unsigned int Value>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::philox_rounds, curanddx::PhiloxRounds<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned int Value>
    struct get_operator_type<curanddx::operator_type, curanddx::PhiloxRounds<Value>> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::philox_rounds;
    };

} // namespace commondx::detail


#endif // CURANDDX_OPERATORS_PHILOX_ROUNDS_HPP
