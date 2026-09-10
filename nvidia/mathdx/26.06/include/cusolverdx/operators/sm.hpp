// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_SM_HPP
#define CUSOLVERDX_OPERATORS_SM_HPP

#include "commondx/operators/sm.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"

namespace cusolverdx {
    // Import SM operator from commonDx
    template <unsigned int Architecture>
    using SM = commondx::SM<Architecture>;
} // namespace cusolverdx

namespace commondx::detail {
    // Thread specializations
    template<unsigned int Architecture>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::sm, cusolverdx::SM<Architecture>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned int Architecture>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::SM<Architecture>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::sm;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_SM_HPP
