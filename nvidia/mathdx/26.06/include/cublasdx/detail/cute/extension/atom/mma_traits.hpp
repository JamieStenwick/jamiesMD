// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_HPP

#include <cute/config.hpp>

#include "cublasdx/detail/system_checks.hpp"

#include "commondx/traits/numeric_traits.hpp"

#include "cublasdx/detail/cute/extension/arch/mma.hpp"

namespace cute {
    template<class A, class B, class C>
    struct MMA_Traits<cublasdx::detail::UniversalFMA<A, B, C>>: MMA_Traits<UniversalFMA<C, A, B, C>> {
    };
} // namespace cute

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_HPP
