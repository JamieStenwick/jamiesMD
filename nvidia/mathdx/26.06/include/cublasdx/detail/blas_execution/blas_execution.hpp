// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_HPP

#include "cublasdx/detail/blas_execution/blas_block_gemm_execution.hpp"
#include "cublasdx/detail/blas_execution/blas_block_trsm_execution.hpp"
#include "cublasdx/detail/blas_execution/blas_thread_trsm_execution.hpp"

namespace cublasdx {
    namespace detail {

        template<int Choice, template<class...> class T1, template<class...> class T2, template<class...> class T3, class... Operators>
        struct get_specialized_execution_type;

        template<template<class...> class T1, template<class...> class T2, template<class...> class T3, class... Operators>
        struct get_specialized_execution_type<0, T1, T2, T3, Operators...> {
            using type = T1<Operators...>;
        };

        template<template<class...> class T1, template<class...> class T2, template<class...> class T3, class... Operators>
        struct get_specialized_execution_type<1, T1, T2, T3, Operators...> {
            using type = T2<Operators...>;
        };

        template<template<class...> class T1, template<class...> class T2, template<class...> class T3, class... Operators>
        struct get_specialized_execution_type<2, T1, T2, T3, Operators...> {
            using type = T3<Operators...>;
        };

        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool is_thread =
                has_operator<operator_type::thread, blas_operator_wrapper<Operators...>>::value;

            static constexpr bool is_block =
                has_operator<operator_type::block, blas_operator_wrapper<Operators...>>::value;

            static constexpr bool is_trsm =
                get_or_default_t<operator_type::function, blas_operator_wrapper<Operators...>, Function<function::MM>>::value == function::TRSM;

            static constexpr bool is_gemm =
                get_or_default_t<operator_type::function, blas_operator_wrapper<Operators...>, Function<function::TRSM>>::value == function::MM;

            static constexpr bool is_ready = (is_thread or is_block) && (is_trsm or is_gemm);

            // Workaround (NVRTC/MSVC)
            //
            // For NVRTC we need to utilize a in-between class called blas_partial_execution_base, otherwise
            // we run into a complation error if Block() is added to description before BLAS description is
            // complete, example:
            //
            // Fails on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + Block() + SM<750>()
            // Works on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + SM<750>() + Block()
            //
            // This workaround disables some useful diagnostics based on static_asserts.
            using specialized_execution_type = typename get_specialized_execution_type<is_thread ? 0 : (is_trsm ? 1 : 2), blas_thread_trsm_execution, blas_block_trsm_execution, blas_block_gemm_execution, Operators...>::type;

            using operator_wrapper_type = blas_operator_wrapper<Operators...>;
            using execution_type = typename COMMONDX_STL_NAMESPACE::conditional<is_complete_blas<operator_wrapper_type>::value and is_ready,
                                                                                specialized_execution_type,
                                                                                blas_partial_execution_base<Operators...>>::type;
            using description_type = blas_description<Operators...>;

        public:
            using type = typename COMMONDX_STL_NAMESPACE::
                conditional<is_ready, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    CUBLASDX_HOST_DEVICE auto operator+(const Operator1&, const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<
            commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
            detail::make_description_t<Operator1, Operator2>>::type {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    CUBLASDX_HOST_DEVICE auto operator+(const detail::blas_description<Operators1...>&,
                                        const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator2>::value,
                                                      detail::make_description_t<Operators1..., Operator2>>::type {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    CUBLASDX_HOST_DEVICE auto operator+(const Operator1&,
                                        const detail::blas_description<Operators2...>&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator1>::value,
                                                      detail::make_description_t<Operator1, Operators2...>>::type {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    CUBLASDX_HOST_DEVICE auto operator+(const detail::blas_description<Operators1...>&,
                                        const detail::blas_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
}
#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_HPP
