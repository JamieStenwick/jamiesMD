// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_INCLUDE_CUSOLVERDX_DETAIL_DESCRIPTION_ARITHMETIC_HPP
#define CUSOLVERDX_INCLUDE_CUSOLVERDX_DETAIL_DESCRIPTION_ARITHMETIC_HPP

#include "cusolverdx/detail/thread_execution.hpp"
#include "cusolverdx/detail/block_execution.hpp"

namespace cusolverdx {
    namespace detail {
        template<class... Operators>
        class cluster_execution;

        template<class... Operators>
        struct make_description {

        private:
            using operator_wrapper_type = solver_operator_wrapper<Operators...>;
            static constexpr bool has_thread_operator     = has_operator<operator_type::thread, operator_wrapper_type>::value;
            static constexpr bool has_block_operator      = has_operator<operator_type::block, operator_wrapper_type>::value;
            static constexpr bool has_cluster_operator    = has_operator<operator_type::cluster, operator_wrapper_type>::value;
            static constexpr bool has_execution_operator  = has_block_operator || has_thread_operator || has_cluster_operator;

            // Workaround (NVRTC/MSVC)
            //
            // For NVRTC we need to utilize a solver_execution class, otherwise
            // we run into a compilation error if Block() is added to description before Solver description is
            // complete, example:
            //
            // Fails on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + Block() + SM<750>()
            // Works on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + SM<750>() + Block()
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
            using cluster_execution_type = typename COMMONDX_STL_NAMESPACE::conditional<is_complete_solver<operator_wrapper_type>::value, cluster_execution<Operators...>, solver_execution<Operators...>>::type;
            using block_execution_type = typename COMMONDX_STL_NAMESPACE::conditional<is_complete_solver<operator_wrapper_type>::value, block_execution<Operators...>, solver_execution<Operators...>>::type;
            using thread_execution_type = typename COMMONDX_STL_NAMESPACE::conditional<is_complete_solver<operator_wrapper_type>::value, thread_execution<Operators...>, solver_execution<Operators...>>::type;
#else
            using cluster_execution_type = cluster_execution<Operators...>;
            using block_execution_type = block_execution<Operators...>;
            using thread_execution_type = thread_execution<Operators...>;
#endif

            using description_type = solver_description<Operators...>;
            using execution_type   = COMMONDX_STL_NAMESPACE::conditional_t<has_block_operator,
                                                              block_execution_type,
                                                              COMMONDX_STL_NAMESPACE::conditional_t<has_thread_operator, thread_execution_type, cluster_execution_type>>;


        public:
            using type = typename COMMONDX_STL_NAMESPACE::conditional<has_execution_operator, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> COMMONDX_STL_NAMESPACE::enable_if_t<commondx::detail::are_operator_expressions<Operator1, Operator2>::value, detail::make_description_t<Operator1, Operator2>> {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::solver_description<Operators1...>&,
                                                       const Operator2&) //
        -> COMMONDX_STL_NAMESPACE::enable_if_t<commondx::detail::is_operator_expression<Operator2>::value, detail::make_description_t<Operators1..., Operator2>> {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::solver_description<Operators2...>&) //
        -> COMMONDX_STL_NAMESPACE::enable_if_t<commondx::detail::is_operator_expression<Operator1>::value, detail::make_description_t<Operator1, Operators2...>> {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::solver_description<Operators1...>&,
                                                       const detail::solver_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cusolverdx
#endif // CUSOLVERDX_INCLUDE_CUSOLVERDX_DETAIL_DESCRIPTION_ARITHMETIC_HPP
