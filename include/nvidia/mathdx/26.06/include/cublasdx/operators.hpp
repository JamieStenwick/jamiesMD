// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_HPP
#define CUBLASDX_OPERATORS_HPP

#include "cublasdx/operators/operator_type.hpp"
#include "cublasdx/operators/alignment.hpp"
#include "cublasdx/operators/arrangement.hpp"
#include "cublasdx/operators/precision.hpp"
#include "cublasdx/operators/size.hpp"
#include "cublasdx/operators/type.hpp"
#include "cublasdx/operators/function.hpp"
#include "cublasdx/operators/ld.hpp"
#include "cublasdx/operators/transpose_mode.hpp"
#include "cublasdx/operators/streaming.hpp"
#include "cublasdx/operators/with_pipeline.hpp"
#include "cublasdx/operators/required_mantissa_bits.hpp"
#include "cublasdx/operators/sm.hpp"
#include "cublasdx/operators/diag.hpp"
#include "cublasdx/operators/side.hpp"
#include "cublasdx/operators/fill_mode.hpp"
#include "cublasdx/operators/batches_per_block.hpp"
#include "commondx/operators/block_dim.hpp"
#include "cublasdx/operators/static_block_dim.hpp"
#include "commondx/operators/execution_operators.hpp"
#include "cublasdx/operators/experimental/tile.hpp"

namespace cublasdx {
    // Import selected operators from commonDx
    struct Block: public commondx::Block {
    };
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: public commondx::BlockDim<X, Y, Z> {
    };
    struct Thread: public commondx::Thread {
    };
} // namespace cublasdx

// Register operators
namespace commondx::detail {
    template<>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::block, cublasdx::Block>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<>
    struct get_operator_type<cublasdx::operator_type, cublasdx::Block> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::block;
    };

    template<>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::thread, cublasdx::Thread>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<>
    struct get_operator_type<cublasdx::operator_type, cublasdx::Thread> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::thread;
    };

    template<unsigned int Architecture>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::sm, cublasdx::SM<Architecture>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int Architecture>
    struct get_operator_type<cublasdx::operator_type, cublasdx::SM<Architecture>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::sm;
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::block_dim, cublasdx::BlockDim<X, Y, Z>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct get_operator_type<cublasdx::operator_type, cublasdx::BlockDim<X, Y, Z>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::block_dim;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_HPP
