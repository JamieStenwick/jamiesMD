// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.


#ifndef CUBLASDX_OPERATORS_BATCHES_PER_BLOCK_HPP
#define CUBLASDX_OPERATORS_BATCHES_PER_BLOCK_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    /// Number of TRSM problem instances (batches) solved per thread block.
    /// When omitted, the library selects a suggested value via the TRSM database.
    template<unsigned int N>
    struct BatchesPerBlock: public commondx::detail::operator_expression {
        static constexpr unsigned int value = N;
    };
} // namespace cublasdx

namespace commondx::detail {
    template<unsigned int N>
    struct is_operator<cublasdx::operator_type,
                       cublasdx::operator_type::batches_per_block,
                       cublasdx::BatchesPerBlock<N>>: COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int N>
    struct get_operator_type<cublasdx::operator_type, cublasdx::BatchesPerBlock<N>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::batches_per_block;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_BATCHES_PER_BLOCK_HPP
