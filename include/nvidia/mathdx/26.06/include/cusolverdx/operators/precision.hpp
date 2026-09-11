// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_PRECISION_HPP
#define CUSOLVERDX_OPERATORS_PRECISION_HPP

#include "commondx/operators/precision.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"
#include "cusolverdx/types.hpp"

namespace cusolverdx {

    template<class T>
    struct PrecisionCheck: public commondx::PrecisionBase<T, __half, __nv_bfloat16, float, double> {};

    template<class PA, class PX = PA, class PB = PA>
    struct Precision: public commondx::detail::operator_expression {
        using a_type = typename PrecisionCheck<PA>::type;
        using x_type = typename PrecisionCheck<PX>::type;
        using b_type = typename PrecisionCheck<PB>::type;
    };

    namespace detail {
        using default_precision_operator = Precision<float, float, float>;
    } // namespace detail
} // namespace cusolverdx

namespace commondx::detail {
    template<class PA, class PX, class PB>
    struct is_operator<cusolverdx::operator_type,
                       cusolverdx::operator_type::precision,
                       cusolverdx::Precision<PA, PX, PB>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<class PA, class PX, class PB>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Precision<PA, PX, PB>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::precision;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_PRECISION_HPP
