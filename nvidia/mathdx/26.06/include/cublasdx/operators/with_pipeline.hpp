// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_WITH_PIPELINE_HPP
#define CUBLASDX_OPERATORS_WITH_PIPELINE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    struct WithPipeline: public commondx::detail::operator_expression {
    };
} // namespace cublasdx

namespace commondx::detail {
    template<>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::with_pipeline, cublasdx::WithPipeline>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<>
    struct get_operator_type<cublasdx::operator_type, cublasdx::WithPipeline> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::with_pipeline;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_WITH_PIPELINE_HPP
