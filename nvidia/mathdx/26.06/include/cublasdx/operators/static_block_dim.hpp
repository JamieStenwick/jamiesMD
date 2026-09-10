// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_STATIC_BLOCKDIM_HPP
#define CUBLASDX_OPERATORS_STATIC_BLOCKDIM_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    struct StaticBlockDim: public commondx::detail::operator_expression {
    };
} // namespace cublasdx

namespace commondx::detail {
    template<>
    struct is_operator<cublasdx::operator_type,
                       cublasdx::operator_type::static_block_dim,
                       cublasdx::StaticBlockDim>: COMMONDX_STL_NAMESPACE::true_type {
    };

    template<>
    struct get_operator_type<cublasdx::operator_type, cublasdx::StaticBlockDim> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::static_block_dim;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_STATIC_BLOCKDIM_HPP
