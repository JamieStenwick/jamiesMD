// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_EXECUTION_HPP
#define CUSOLVERDX_OPERATORS_EXECUTION_HPP

#include "commondx/operators/execution_operators.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"

namespace cusolverdx {
    // Import selected operators from commonDx
    struct Thread: public commondx::Thread {};

    // Import Execution operators from commonDx
    struct Block: public commondx::Block {};

    struct Cluster: public commondx::Cluster {};

} // namespace cusolverdx

namespace commondx::detail {
    // Thread specializations
    template<>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::thread, cusolverdx::Thread>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Thread> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::thread;
    };
    // Block specializations
    template<>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::block, cusolverdx::Block>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Block> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::block;
    };
    // Cluster specializations
    template<>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::cluster, cusolverdx::Cluster>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Cluster> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::cluster;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_EXECUTION_HPP
