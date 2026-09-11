// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_BATCHES_PER_BLOCK_HPP
#define CUSOLVERDX_OPERATORS_BATCHES_PER_BLOCK_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cusolverdx {

    template<unsigned Value>
    struct BatchesPerBlock : public commondx::detail::constant_operator_expression<unsigned, Value> {
    };

    namespace detail {
        using default_batches_per_block_operator = BatchesPerBlock<1>;
    } // namespace detail
} // namespace cusolverdx

// Register operators
namespace commondx::detail {
    template<unsigned Value>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::batches_per_block, cusolverdx::BatchesPerBlock<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned Value>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::BatchesPerBlock<Value>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::batches_per_block;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_BATCHES_PER_BLOCK_HPP
