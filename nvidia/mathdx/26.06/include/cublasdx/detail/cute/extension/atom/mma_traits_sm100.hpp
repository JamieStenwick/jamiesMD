// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM100_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM100_HPP

#include "cublasdx/detail/cute/extension/arch/mma_sm100.hpp"

namespace cute {

    template<>
    struct MMA_Traits<cublasdx::detail::SM100_2x1x1_F32F32F32F32> {
        using ValTypeD = float;
        using ValTypeA = float;
        using ValTypeB = float;
        using ValTypeC = float;

        using Shape_MNK = Shape<_2, _1, _1>;
        using ThrID     = Layout<_1>;

        using ALayout = Layout<Shape<_1, _2>>;
        using BLayout = Layout<Shape<_1, _1>>;
        using CLayout = Layout<Shape<_1, _2>>;
    };

    template<>
    struct MMA_Traits<cublasdx::detail::SM100_1x2x1_F32F32F32F32> {
        using ValTypeD = float;
        using ValTypeA = float;
        using ValTypeB = float;
        using ValTypeC = float;

        using Shape_MNK = Shape<_1, _2, _1>;
        using ThrID     = Layout<_1>;

        using ALayout = Layout<Shape<_1, _1>>;
        using BLayout = Layout<Shape<_1, _2>>;
        using CLayout = Layout<Shape<_1, _2>>;
    };

} // namespace cute

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM100_HPP
