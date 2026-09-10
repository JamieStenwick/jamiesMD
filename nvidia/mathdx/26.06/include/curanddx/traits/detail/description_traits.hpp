// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
#define CURANDDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/description_traits.hpp"

namespace curanddx {
    namespace detail {
        template<operator_type OperatorType, class Description>
        using get_t = commondx::detail::get_t<operator_type, OperatorType, Description>;

        template<operator_type OperatorType, class Description, class Default = void>
        using get_or_default_t = commondx::detail::get_or_default_t<operator_type, OperatorType, Description, Default>;

        template<operator_type OperatorType, class Description>
        using has_operator = commondx::detail::has_operator<operator_type, OperatorType, Description>;

        template<operator_type OperatorType, class Description>
        using has_at_most_one_of = commondx::detail::has_at_most_one_of<operator_type, OperatorType, Description>;

        template<unsigned int N, operator_type OperatorType, class Description>
        using has_n_of = commondx::detail::has_n_of<N, operator_type, OperatorType, Description>;

        // is_complete_description
        namespace is_complete_description_impl {

            template<class Description, class Enable = void>
            struct helper: COMMONDX_STL_NAMESPACE::false_type {};

            template<template<class...> class Description, class... Types>
            struct helper<Description<Types...>,
                          typename COMMONDX_STL_NAMESPACE::enable_if<
                              commondx::detail::is_description_expression<Description<Types...>>::value>::type> {
                using description_type = Description<Types...>;

                // Generator
                using my_generator =
                    get_or_default_t<operator_type::generator, description_type, default_curanddx_generator_operator>;
                // SM
                using my_sm = get_t<operator_type::sm, description_type>;

                // Thread
                static constexpr bool is_thread_execution = has_operator<operator_type::thread, description_type>::value;

                static constexpr bool value =
                    !(COMMONDX_STL_NAMESPACE::is_void<my_generator>::value ||
                    // for thread execution, we don't require SM for completeness
                    (COMMONDX_STL_NAMESPACE::is_void<my_sm>::value && !is_thread_execution));
            };
        } // namespace is_complete_description_impl

        template<class Description>
        struct is_complete_description:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, is_complete_description_impl::helper<Description>::value> {};
    } // namespace detail
} // namespace curanddx

#endif // CURANDDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
