// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_FUNCTION_HPP
#define CUSOLVERDX_OPERATORS_FUNCTION_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx {

    template<function Value>
    struct Function: commondx::detail::operator_expression {
        static_assert((Value == potrf) || (Value == potrs) || (Value == posv) || (Value == getrf_no_pivot) || (Value == getrs_no_pivot) || (Value == gesv_no_pivot) || (Value == getrf_partial_pivot) ||
                      (Value == getrs_partial_pivot) || (Value == gesv_partial_pivot) || (Value == gtsv_no_pivot) || (Value == modified_lu) ||
                      (Value == geqrf) || (Value == unmqr) ||(Value == gelqf) || (Value == unmlq) || (Value == gels) ||
                      (Value == ungqr) || (Value == unglq) ||
                      (Value == htev) || (Value == heev) || (Value == hegst) || (Value == hegv) ||
                      (Value == bdsvd) || (Value == gesvd) ||
                      (Value == trsm),
                      "Unsupported function");

        static constexpr function value = Value;
    };

    namespace detail {
        using default_function_operator = Function<potrf>;

        template<function Value>
        struct is_cholesky: COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == potrf || Value == potrs || Value == posv)> {};

        template<function Value>
        struct is_lu_no_pivot:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == getrf_no_pivot || Value == getrs_no_pivot || Value == gesv_no_pivot || Value == gtsv_no_pivot || Value == modified_lu)> {};

        template<function Value>
        struct is_lu_partial_pivot:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == getrf_partial_pivot || Value == getrs_partial_pivot || Value == gesv_partial_pivot)> {};

        template<function Value>
        struct is_gtsv_no_pivot:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == gtsv_no_pivot)> {};

        template<function Value>
        struct is_lu:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, is_lu_no_pivot<Value>::value || is_lu_partial_pivot<Value>::value> {};

        template<function Value>
        struct is_qr: COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == geqrf) || (Value == unmqr) || (Value == gelqf) || (Value == unmlq) ||(Value == gels) || (Value == ungqr) || (Value == unglq)> {};

        template<function Value>
        struct is_unmq: COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == unmlq) ||(Value == unmqr)> {};

        template<function Value>
        struct is_ungq: COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == ungqr) || (Value == unglq)> {};

        template<function Value>
        struct is_trsm: COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == trsm)> {};

        template<function Value>
        struct is_symmetric_eigen: COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == htev) || (Value == heev) || (Value == hegst) || (Value == hegv)> {};

        template<function Value>
        struct is_svd: COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == bdsvd) || (Value == gesvd)> {};

        template<function Value>
        struct has_both_a_and_b_matrices: 
            COMMONDX_STL_NAMESPACE::integral_constant<bool,
                                                      (Value == potrs || Value == posv || Value == getrs_no_pivot || Value == gesv_no_pivot || Value == getrs_partial_pivot ||
                                                       Value == gesv_partial_pivot || Value == gtsv_no_pivot ||
                                                       Value == gels || Value == unmqr || Value == unmlq || Value == hegst || Value == hegv ||
                                                       Value == trsm)> {};


    } // namespace detail
} // namespace cusolverdx

namespace commondx::detail {
    template<cusolverdx::function Value>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::function, cusolverdx::Function<Value>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::function Value>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Function<Value>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::function;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_FUNCTION_HPP
