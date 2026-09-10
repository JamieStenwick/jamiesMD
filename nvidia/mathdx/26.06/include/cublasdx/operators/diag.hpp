// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_DIAG_HPP
#define CUBLASDX_OPERATORS_DIAG_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    enum class diag
    {
        non_unit,
        unit,
    };

    template<diag Value>
    struct Diag: public commondx::detail::constant_operator_expression<diag, Value> {
    };
} // namespace cublasdx

namespace commondx::detail {
    template<cublasdx::diag Value>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::diag, cublasdx::Diag<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cublasdx::diag Value>
    struct get_operator_type<cublasdx::operator_type, cublasdx::Diag<Value>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::diag;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_DIAG_HPP
