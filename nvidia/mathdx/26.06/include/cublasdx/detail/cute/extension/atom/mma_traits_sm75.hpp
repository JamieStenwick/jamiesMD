// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM75_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM75_HPP

#include <cute/config.hpp>

#include "cublasdx/detail/system_checks.hpp"

#include "commondx/traits/numeric_traits.hpp"

#include "cublasdx/detail/cute/extension/arch/mma_sm75.hpp"

namespace cute {
    template<>
    struct MMA_Traits<cublasdx::detail::SM75_16x8x8_F16F16F16F16_TN> {
        using ValTypeD = half_t;
        using ValTypeA = half_t;
        using ValTypeB = half_t;
        using ValTypeC = half_t;

        using Shape_MNK = Shape<_16, _8, _8>;
        using ThrID     = Layout<_32>;
        using ALayout   = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>, Stride<Stride<_32, _1>, Stride<_16, _8>>>;
        using BLayout   = Layout<Shape<Shape<_4, _8>, _2>, Stride<Stride<_16, _1>, _8>>;
        using CLayout   = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>, Stride<Stride<_32, _1>, Stride<_16, _8>>>;
    };

    template<>
    struct MMA_Traits<cublasdx::detail::SM75_16x8x8_F32F16F16F32_TN>:
        MMA_Traits<cublasdx::detail::SM75_16x8x8_F16F16F16F16_TN> {
        using ValTypeD = float;
        using ValTypeA = half_t;
        using ValTypeB = half_t;
        using ValTypeC = float;
    };

} // namespace cute

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM75_HPP
