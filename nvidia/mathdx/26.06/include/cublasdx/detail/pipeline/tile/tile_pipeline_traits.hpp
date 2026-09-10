// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_PIPELINE_TRAITS_HPP
#define CUBLASDX_DETAIL_PIPELINE_PIPELINE_TRAITS_HPP

#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/detail/pipeline/accumulator_mode.hpp"

namespace cublasdx {
    namespace detail {

        template<class T>
        struct is_pipeline: cute::false_type {
        };

        template<class T>
        struct is_tile_pipeline: is_pipeline<T> {
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_PIPELINE_TRAITS_HPP
