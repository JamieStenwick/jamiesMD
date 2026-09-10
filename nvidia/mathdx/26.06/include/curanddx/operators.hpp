// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_OPERATORS_HPP
#define CURANDDX_OPERATORS_HPP

#include "curanddx/operators/operator_type.hpp"
#include "curanddx/operators/generator.hpp"
#include "curanddx/operators/ordering.hpp"
#include "curanddx/operators/philox_rounds.hpp"

#include "commondx/operators/block_dim.hpp"
#include "commondx/operators/sm.hpp"
#include "commondx/operators/execution_operators.hpp"

namespace curanddx {
    // Import selected operators from commonDx
    struct Thread: public commondx::Thread {
    };
    struct Block: public commondx::Block {
    };
    struct Device: public commondx::Device {
    };
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: public commondx::BlockDim<X, Y, Z> {};
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct GridDim: public commondx::GridDim<X, Y, Z> {};
    template <unsigned int Architecture>
    using SM = commondx::SM<Architecture>;
} // namespace curanddx

// Register operators
namespace commondx::detail {
    template<>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::thread, curanddx::Thread>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<>
    struct get_operator_type<curanddx::operator_type, curanddx::Thread> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::thread;
    };

    template<>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::block, curanddx::Block>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<>
    struct get_operator_type<curanddx::operator_type, curanddx::Block> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::block;
    };

    template<>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::device, curanddx::Device>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<>
    struct get_operator_type<curanddx::operator_type, curanddx::Device> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::device;
    };

    template<unsigned int Architecture>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::sm, curanddx::SM<Architecture>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned int Architecture>
    struct get_operator_type<curanddx::operator_type, curanddx::SM<Architecture>> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::sm;
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::block_dim, curanddx::BlockDim<X, Y, Z>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct get_operator_type<curanddx::operator_type, curanddx::BlockDim<X, Y, Z>> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::block_dim;
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::grid_dim, curanddx::GridDim<X, Y, Z>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct get_operator_type<curanddx::operator_type, curanddx::GridDim<X, Y, Z>> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::grid_dim;
    };
} // namespace commondx::detail

#endif // CURANDDX_OPERATORS_HPP
