// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_TRAITS_DETAIL_IS_COMPLETE_HPP
#define CUSOLVERDX_TRAITS_DETAIL_IS_COMPLETE_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/description_traits.hpp"

namespace cusolverdx {
    namespace detail {
        // is_complete_description
        namespace is_complete_description_impl {
            template<class Description, class Enable = void>
            struct helper: COMMONDX_STL_NAMESPACE::false_type {};

            template<template<class...> class Description, class... Types>
            struct helper<Description<Types...>,
                          COMMONDX_STL_NAMESPACE::enable_if_t<
                              commondx::detail::is_description_expression<Description<Types...>>::value>> {
                using description_type = Description<Types...>;

                // ---------
                // Mandatory
                // ---------
                // SM
                static constexpr bool has_sm = has_operator_v<operator_type::sm, description_type>;
                // Size
                static constexpr bool has_size = has_operator_v<operator_type::size, description_type>;
                // Function
                static constexpr bool has_function = has_operator_v<operator_type::function, description_type>;

                static constexpr bool value = has_sm and has_size and has_function;
            };
        } // namespace is_complete_description_impl

        // is_execution_description
        namespace is_execution_description_impl {
            template<class Description, class Enable = void>
            struct helper: COMMONDX_STL_NAMESPACE::false_type {};

            template<template<class...> class Description, class... Types>
            struct helper<Description<Types...>,
                          COMMONDX_STL_NAMESPACE::enable_if_t<
                              commondx::detail::is_description_expression<Description<Types...>>::value>> {
                using description_type = Description<Types...>;

                // ---------
                // Mandatory
                // ---------
                // Execution
                static constexpr bool has_block   = has_operator_v<operator_type::block, description_type>;
                static constexpr bool has_thread  = has_operator_v<operator_type::thread, description_type>;
                static constexpr bool has_cluster = has_operator_v<operator_type::cluster, description_type>;

                static constexpr bool value = has_block || has_thread || has_cluster;
            };
        } // namespace is_execution_description_impl

        template<class Description>
        struct is_complete_description:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, is_complete_description_impl::helper<Description>::value> {
        };

        template<class Description>
        struct is_complete_execution_description:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, is_execution_description_impl::helper<Description>::value &&
              is_complete_description_impl::helper<Description>::value> {
        };

    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_TRAITS_DETAIL_IS_COMPLETE_HPP
