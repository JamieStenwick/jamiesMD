// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_BLOCKDIM_HPP
#define CUSOLVERDX_OPERATORS_BLOCKDIM_HPP

#include "commondx/operators/block_dim.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"

namespace cusolverdx {
    // Import BlockDim operator from commonDx
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: public commondx::BlockDim<X, Y, Z> {
        static_assert(Y != 1 or Z == 1, "Threads in BlockDim must be contiguous, BlockDim<X, 1, Z> is incorrect");
    };
} // namespace cusolverdx

namespace commondx::detail {
    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::block_dim, cusolverdx::BlockDim<X, Y, Z>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::BlockDim<X, Y, Z>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::block_dim;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_BLOCKDIM_HPP
