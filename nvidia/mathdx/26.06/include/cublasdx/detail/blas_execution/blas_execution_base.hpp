// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_BASE_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_BASE_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "cublasdx/database/gemm/suggested/suggested_layouts.hpp"
#include "cublasdx/database/trsm/trsm_execute.hpp"

#include "cublasdx/detail/blas_description.hpp"
#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/detail/blas_backend.hpp"
#include "cublasdx/traits.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_traits.hpp"
#include "cublasdx/detail/functional.hpp"

#include "cublasdx/detail/blas_execution_bugs.hpp"

namespace cublasdx {
    namespace detail {
        enum matrix
        {
            A = 0,
            B = 1,
            C = 2
        };

        template<class InType, class ExpectedType>
        constexpr bool is_type_compatible() {
            return (sizeof(InType) == sizeof(ExpectedType)) && (alignof(InType) == alignof(ExpectedType));
        }

        template<class TypeTuple>
        static constexpr bool CUBLASDX_HOST_DEVICE is_type_pair_compatible() {
            using TA = typename COMMONDX_STL_NAMESPACE::tuple_element<0, TypeTuple>::type;
            using TB = typename COMMONDX_STL_NAMESPACE::tuple_element<1, TypeTuple>::type;

            // sizeof(void) is illegal, so it needs to be avoided
            if constexpr (COMMONDX_STL_NAMESPACE::is_void_v<TA> or (COMMONDX_STL_NAMESPACE::is_void_v<TB>)) {
                return false;
            } else {
                return ((sizeof(TA) == sizeof(TB)) && (alignof(TA) == alignof(TB))) || cute::is_convertible_v<TB, TA>;
            }

            CUTE_GCC_UNREACHABLE;
        }

        template<typename... TypeTuples>
        static constexpr bool CUBLASDX_HOST_DEVICE are_types_compatible_impl() {
            return (is_type_pair_compatible<TypeTuples>() && ...);
        }

        template<class value_type>
        struct cutlass_value_type {
            using a_type = convert_to_cutlass_type_t<typename value_type::a_type>;
            using b_type = convert_to_cutlass_type_t<typename value_type::b_type>;
            using c_type = convert_to_cutlass_type_t<typename value_type::c_type>;
        };

        template<class SpecializedExecutionType, class... Operators>
        class blas_execution_base:
            public blas_description<Operators...>,
            public commondx::detail::execution_description_expression
        {
            using base_type = blas_description<Operators...>;
            using this_type = blas_execution_base<SpecializedExecutionType, Operators...>;

        protected:
            // Precision type
            using typename base_type::this_blas_precision;

            /// ---- Constraints

            // At most one execution-mode operator is allowed.
            static constexpr bool has_at_most_one_block  = has_at_most_one_of<operator_type::block,  this_type>::value;
            static constexpr bool has_at_most_one_thread = has_at_most_one_of<operator_type::thread, this_type>::value;
            static_assert(has_at_most_one_block,  "Can't create blas function with two Block operators");
            static_assert(has_at_most_one_thread, "Can't create blas function with two Thread operators");
            
            static constexpr bool has_block     = has_operator<operator_type::block,     base_type>::value;
            static constexpr bool has_thread    = has_operator<operator_type::thread,    base_type>::value;

            // TRSM: batches per block - user-provided or database suggestion.
            static constexpr bool is_trsm = (base_type::this_blas_function_v == function::TRSM);
            static constexpr bool is_gemm = (base_type::this_blas_function_v == function::MM);            

            // GCC7 workaround
            using this_blas_size = typename base_type::this_blas_size;

            // Value type
            using this_blas_value_type    = map_value_type<base_type::this_blas_type_v, this_blas_precision>;
            static constexpr bool has_ld = has_operator<operator_type::ld, base_type>::value;

            
        public:

            // Pointer alignments of A, B, C matrices in bytes
            static constexpr unsigned int a_alignment = base_type::this_blas_alignment_a;
            static constexpr unsigned int b_alignment = base_type::this_blas_alignment_b;
            static constexpr unsigned int c_alignment = base_type::this_blas_alignment_c;

            using a_value_type = typename this_blas_value_type::a_type;
            using b_value_type = typename this_blas_value_type::b_type;
            using c_value_type = typename this_blas_value_type::c_type;

            template<typename... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto get_layout_gmem_a(Ts... ts) {
                return SpecializedExecutionType::template tag_cute_layout<matrix::A, commondx::detail::gmem_tag>(ts...);
            }

            template<typename... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto get_layout_gmem_b(Ts... ts) {
                return SpecializedExecutionType::template tag_cute_layout<matrix::B, commondx::detail::gmem_tag>(ts...);
            }

            template<typename... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto get_layout_gmem_c(Ts... ts) {
                return SpecializedExecutionType::template tag_cute_layout<matrix::C, commondx::detail::gmem_tag>(ts...);
            }

            template<typename... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto get_layout_smem_a(Ts... ts) {
                return SpecializedExecutionType::template tag_cute_layout<matrix::A, commondx::detail::smem_tag>(ts...);
            }

            template<typename... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto get_layout_smem_b(Ts... ts) {
                return SpecializedExecutionType::template tag_cute_layout<matrix::B, commondx::detail::smem_tag>(ts...);
            }

            template<typename... Ts>
            CUBLASDX_HOST_DEVICE constexpr static auto get_layout_smem_c(Ts... ts) {
                return SpecializedExecutionType::template tag_cute_layout<matrix::C, commondx::detail::smem_tag>(ts...);
            }
        };


        template<class... Operators>
        class blas_partial_execution_base: public blas_execution_base<blas_partial_execution_base<Operators...>, Operators...>
        {
            using base_type = blas_execution_base<blas_partial_execution_base<Operators...>, Operators...>;
            using typename base_type::this_blas_precision;
            using this_blas_value_type = map_value_type<base_type::this_blas_type_v, this_blas_precision>;

        public:
            using a_value_type = typename this_blas_value_type::a_type;
            using b_value_type = typename this_blas_value_type::b_type;
            using c_value_type = typename this_blas_value_type::c_type;
        };
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_BASE_HPP
