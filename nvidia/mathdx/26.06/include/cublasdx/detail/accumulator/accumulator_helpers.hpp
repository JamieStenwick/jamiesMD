// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_ACCUMULATOR_ACCUMULATOR_HELPERS_HPP
#define CUBLASDX_DETAIL_ACCUMULATOR_ACCUMULATOR_HELPERS_HPP

#include "cublasdx/detail/functional.hpp"

namespace cublasdx {

    using cute::clear;
    using cute::make_coord;
    using cute::make_fragment_like;

    template<class... Args>
    CUBLASDX_HOST_DEVICE void transform_fragment(Args&&... args) {
        cute::transform(static_cast<Args&&>(args)...);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_ACCUMULATOR_ACCUMULATOR_HELPERS_HPP
