// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_DIAG_HPP
#define CUSOLVERDX_OPERATORS_DIAG_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx {

    template<diag d>
    struct Diag: commondx::detail::operator_expression {
        static_assert(d == diag::non_unit || d == diag::unit, "Diag has to be non_unit or unit");

        static constexpr diag value = d;
    };

    inline constexpr auto non_unit = diag::non_unit;
    inline constexpr auto unit     = diag::unit;

} // namespace cusolverdx

namespace commondx::detail {
    template<cusolverdx::diag diag>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::diag, cusolverdx::Diag<diag>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::diag diag>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Diag<diag>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::diag;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_DIAG_HPP
