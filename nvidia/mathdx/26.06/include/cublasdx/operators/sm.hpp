// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_SM_HPP
#define CUBLASDX_OPERATORS_SM_HPP

#include "commondx/operators/sm.hpp"

namespace cublasdx {

    enum class sm_modifier
    {
        generic,
        arch_specific,
        family_specific
    };

    inline constexpr sm_modifier generic         = sm_modifier::generic;
    inline constexpr sm_modifier arch_specific   = sm_modifier::arch_specific;
    inline constexpr sm_modifier family_specific = sm_modifier::family_specific;

    template<unsigned int Architecture, sm_modifier Modifier = generic>
    struct SM: commondx::SM<Architecture> {
        static constexpr sm_modifier modifier = Modifier;
        static_assert(Modifier == generic or Architecture >= 900, "SM modifiers other than generic are only supported for SM >= 900");
    };
} // namespace cublasdx


namespace commondx::detail {
    template<unsigned int Architecture, cublasdx::sm_modifier Modifier>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::sm, cublasdx::SM<Architecture, Modifier>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int Architecture, cublasdx::sm_modifier Modifier>
    struct get_operator_type<cublasdx::operator_type, cublasdx::SM<Architecture, Modifier>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::sm;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_SM_HPP
