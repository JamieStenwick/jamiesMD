// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_ARRANGEMENT_HPP
#define CUSOLVERDX_OPERATORS_ARRANGEMENT_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx {

    template<arrangement Arr, arrangement Brr = Arr, arrangement Crr = Arr> //Crr only used for gesvd with vectors
    struct Arrangement: commondx::detail::operator_expression {
        static_assert((Arr == arrangement::col_major || Arr == arrangement::row_major) && (Brr == arrangement::row_major || Brr == arrangement::col_major) &&
                      (Crr == arrangement::col_major || Crr == arrangement::row_major), "Arrangement has to be col_major or row_major");

        static constexpr arrangement a = Arr;
        static constexpr arrangement b = Brr;
        static constexpr arrangement c = Crr;
    };

    namespace detail {
        using default_arrangement_operator = Arrangement<arrangement::col_major>;
    } // namespace detail
} // namespace cusolverdx

namespace commondx::detail {
    template<cusolverdx::arrangement Arr, cusolverdx::arrangement Brr, cusolverdx::arrangement Crr>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::arrangement, cusolverdx::Arrangement<Arr, Brr, Crr>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::arrangement Arr, cusolverdx::arrangement Brr, cusolverdx::arrangement Crr>
    struct get_operator_type<cusolverdx::operator_type,cusolverdx::Arrangement<Arr, Brr, Crr>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::arrangement;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_ARRANGEMENT_HPP
