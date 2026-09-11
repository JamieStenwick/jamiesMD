// Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_LEADING_DIMENSION_HPP
#define CUSOLVERDX_OPERATORS_LEADING_DIMENSION_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cusolverdx {
    template<unsigned int LDA, unsigned int LDB = LDA, unsigned int LDC = 0> //LDC only used for gesvd with vectors
    struct LeadingDimension: public commondx::detail::operator_expression {
        static constexpr unsigned int a = LDA;
        static constexpr unsigned int b = LDB;
        static constexpr unsigned int c = LDC;
    };
} // namespace cusolverdx


namespace commondx::detail {
    template<unsigned int LDA, unsigned int LDB, unsigned int LDC> 
    struct is_operator<cusolverdx::operator_type,
                       cusolverdx::operator_type::leading_dimension,
                       cusolverdx::LeadingDimension<LDA, LDB, LDC>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned int LDA, unsigned int LDB, unsigned int LDC> 
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::LeadingDimension<LDA, LDB, LDC>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::leading_dimension;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_LEADING_DIMENSION_HPP
