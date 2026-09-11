// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_REQUIRED_MANTISSA_BITS_HPP
#define CUBLASDX_OPERATORS_REQUIRED_MANTISSA_BITS_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cublasdx/operators/operator_type.hpp"

namespace cublasdx {
    template<unsigned int X>
    struct RequiredMantissaBits: public commondx::detail::operator_expression {
        static_assert(X > 0, "RequiredMantissaBits<X> requires X > 0");
        static constexpr unsigned int value = X;
    };
} // namespace cublasdx

namespace commondx::detail {
    template<unsigned int X>
    struct is_operator<cublasdx::operator_type,
                       cublasdx::operator_type::required_mantissa_bits,
                       cublasdx::RequiredMantissaBits<X>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int X>
    struct get_operator_type<cublasdx::operator_type, cublasdx::RequiredMantissaBits<X>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::required_mantissa_bits;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_REQUIRED_MANTISSA_BITS_HPP
