// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_BLOCK_TRSM_EXECUTION_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_BLOCK_TRSM_EXECUTION_HPP

#include "cublasdx/detail/blas_execution/blas_execution_base.hpp"

namespace cublasdx::detail {

    template<class... Operators>
    class blas_block_trsm_execution : public blas_execution_base<blas_block_trsm_execution<Operators...>, Operators...> {
        using this_type = blas_block_trsm_execution<Operators...>;
        using base_type = blas_execution_base<this_type, Operators...>;

        /// ---- Traits
        // Imported shared types
        using typename base_type::this_blas_value_type;
        using typename base_type::this_blas_precision;
        using typename base_type::this_blas_size;

        public:
        using typename base_type::a_value_type;
        using typename base_type::b_value_type;

        using base_type::a_alignment;
        using base_type::b_alignment;
        
        static constexpr int lda = base_type::this_blas_trsm_lda;
        static constexpr int ldb = base_type::this_blas_trsm_ldb;

        static constexpr auto a_shape = cute::make_shape(cute::Int<base_type::this_blas_trsm_dim_a> {},
                                                        cute::Int<base_type::this_blas_trsm_dim_a> {});
        using a_shape_t               = decltype(a_shape);
        static constexpr auto b_shape = cute::make_shape(cute::Int<base_type::this_blas_size_m_v> {},
                                                         cute::Int<base_type::this_blas_size_n_v> {});
        using b_shape_t               = decltype(b_shape);

        protected:

        // Imported shared constants
        using base_type::has_ld;

        static constexpr int sm_v = base_type::this_blas_sm_v;

        template<matrix M, class MemTag>
        static constexpr int get_default_ld() {
            // TRSM smem: A is dim_axdim_a, B is MxN.
            if (M == matrix::A) {
                return static_cast<int>(base_type::this_blas_trsm_lda);
            } 
            
            return static_cast<int>(base_type::this_blas_trsm_ldb);
        }

        template<matrix M, class MemTag, class LDInner = cute::Int<get_default_ld<M, MemTag>()>>
        CUBLASDX_HOST_DEVICE constexpr static auto get_inner_layout(LDInner ld_inner = {}) {
            if constexpr (M == matrix::A) {
                constexpr unsigned    dim_a = base_type::this_blas_trsm_dim_a;
                constexpr arrangement arr   = base_type::this_blas_arrangement_a;
                return cute_backend::make_layout_from_arrangement<arr>(
                    cute::Int<dim_a> {}, cute::Int<dim_a> {}, ld_inner);
            } else {
                constexpr arrangement arr = base_type::this_blas_arrangement_b;
                return cute_backend::make_layout_from_arrangement<arr>(
                    cute::Int<this_blas_size::m> {},
                    cute::Int<this_blas_size::n> {},
                    ld_inner);
            }

            CUTE_GCC_UNREACHABLE;
        }

        // When 0 or 1 arguments are specified
        template<matrix M, class MemTag, class LDInner = cute::Int<get_default_ld<M, MemTag>()>>
        CUBLASDX_HOST_DEVICE constexpr static auto get_cute_layout(LDInner ld_inner = {}) {
            static_assert(cute::is_integral<LDInner>::value);

            // TRSM per-batch tile layout: A is dim_axdim_a (square), B is MxN.
            auto inner = get_inner_layout<M, MemTag>(ld_inner);

            if constexpr (batches_per_block > 1) {
                return cute::append<3>(inner, cute::make_layout(cute::Int<batches_per_block>{}, cublasdx::cosize(inner)));
            } else {
                return inner;
            }
            CUTE_GCC_UNREACHABLE;
        }

        template<matrix M, class MemTag, class LDInner, class LDOuter>
        CUBLASDX_HOST_DEVICE constexpr static auto get_cute_layout(LDInner ld_inner = {}, LDOuter ld_outer = {}) {
            static_assert(batches_per_block > 1, "Batches per block must be greater than 1 if BatchLD is specified");
            auto inner = get_inner_layout<M, MemTag>(ld_inner);
            return cute::flatten(cute::make_layout(inner, cute::make_layout(cute::Int<batches_per_block>{}, ld_outer)));
        }

        public:

        template<matrix M, class MemTag, class ... Args>
        CUBLASDX_HOST_DEVICE constexpr static auto tag_cute_layout(Args... args) {
            return commondx::detail::pointer_layout {MemTag {}, get_cute_layout<M, MemTag>(args...)};
        }

        static constexpr unsigned int a_size = base_type::this_blas_trsm_a_size;
        static constexpr unsigned int b_size = base_type::this_blas_trsm_b_size;

        static constexpr unsigned suggested_batches_per_block =
            trsm::suggested_batches<typename this_blas_value_type::a_type,
                                           base_type::this_blas_size_m_v,
                                           base_type::this_blas_size_n_v,
                                           (base_type::this_blas_side_v == side::left),
                                           base_type::this_blas_sm_v>();

        static constexpr unsigned batches_per_block =
            base_type::has_batches_per_block
                ? get_or_default_t<operator_type::batches_per_block, this_type,
                             BatchesPerBlock<1>>::value
            : 1u;

        static constexpr CUBLASDX_HOST_DEVICE dim3 get_suggested_block_dim() {
            static_assert(base_type::is_complete,
                            "Can't provide suggested block dimensions, description is not complete");
            return trsm::suggested_block_dim<typename this_blas_value_type::a_type, 
                                             (base_type::this_blas_side_v == side::left 
                                                ? base_type::this_blas_size_m_v 
                                                : base_type::this_blas_size_n_v), 
                                              batches_per_block, base_type::this_blas_sm_v>();
        }

        static constexpr CUBLASDX_HOST_DEVICE dim3 get_block_dim() {
            static_assert(base_type::is_complete, "Can't provide block dimensions, description is not complete");
            if constexpr (base_type::has_block_dim) {
                return base_type::this_blas_block_dim_v;
            }
            return get_suggested_block_dim();
        }

        static constexpr dim3 suggested_block_dim = this_type::get_suggested_block_dim();
        static constexpr dim3 block_dim = this_type::get_block_dim();

        static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;

        // ---- TRSM execute -----------------------------------------------
        // Solves the TRSM problem in-place on tensor_b.
        // Any CuTe-compatible layout may be used; tensors must have the correct
        // shape for the configured Size and Side. get_layout_smem_a/b() provide
        // convenient defaults for contiguous shared-memory allocation.
        // For BPB > 1 the tensors must carry a batch mode as their third index
        // (use a layout built from get_layout_smem_* extended with a batch
        // stride of a_size / b_size).

        // Tensor API
        template<class AEngine,
                 class ALayout,
                 class BEngine,
                 class BLayout>
        CUBLASDX_DEVICE  
        cute::enable_if_t<is_type_compatible<typename AEngine::value_type, typename base_type::a_value_type>() and
                          is_type_compatible<typename BEngine::value_type, typename base_type::b_value_type>()>
        execute(const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                cublasdx::tensor<BEngine, BLayout>&       tensor_b) const {

            auto input_a = safe_recast<typename base_type::a_value_type>(tensor_a);
            auto input_b = safe_recast<typename base_type::b_value_type>(tensor_b);

            trsm::block_execute<base_type::this_blas_size_m_v,
                                base_type::this_blas_size_n_v,
                                base_type::this_blas_side_v,
                                base_type::this_blas_diag_v,
                                base_type::this_blas_fill_mode_v,
                                get_threads<BlockDim<block_dim.x, block_dim.y, block_dim.z>>(),
                                batches_per_block,
                                base_type::this_blas_alignment_b>(input_a, input_b,
                                    get_thread_idx<BlockDim<block_dim.x, block_dim.y, block_dim.z>>());
        }

        // Pointer API, only pointers or only LDB
        template<class TA, class TB, class LDB = cute::Int<base_type::this_blas_trsm_ldb>>
        CUBLASDX_DEVICE 
        cute::enable_if_t<is_type_compatible<TA, typename base_type::a_value_type>() and
                          is_type_compatible<TB, typename base_type::b_value_type>()>
        execute(TA const* matrix_a, TB * matrix_b, LDB ldb = {}) const {
            auto a_tensor = cublasdx::make_tensor(matrix_a, this_type::get_layout_smem_a());
            auto b_tensor = cublasdx::make_tensor(matrix_b, this_type::get_layout_smem_b(ldb));
            this_type::execute(a_tensor, b_tensor);
        }

        // Pointer API, only LDA or (LDA and LDB)
        template<class TA,
                 class LDA,
                 class TB,
                 class LDB = cute::Int<base_type::this_blas_trsm_ldb>>
        CUBLASDX_DEVICE 
        cute::enable_if_t<is_type_compatible<TA, typename base_type::a_value_type>() and
                          is_type_compatible<TB, typename base_type::b_value_type>()>
        execute(TA const* matrix_a, LDA lda, TB * matrix_b, LDB ldb = {}) const {
            auto a_tensor = cublasdx::make_tensor(matrix_a, this_type::get_layout_smem_a(lda));
            auto b_tensor = cublasdx::make_tensor(matrix_b, this_type::get_layout_smem_b(ldb));
            this_type::execute(a_tensor, b_tensor);
        }

        // Fallbacks for static_asserts
        template<class AEngine,
                 class ALayout,
                 class BEngine,
                 class BLayout>
        CUBLASDX_DEVICE  
        cute::enable_if_t<not is_type_compatible<typename AEngine::value_type, typename base_type::a_value_type>() or
                          not is_type_compatible<typename BEngine::value_type, typename base_type::b_value_type>()>
        execute(const cublasdx::tensor<AEngine, ALayout>& /* tensor_a */,
                cublasdx::tensor<BEngine, BLayout>&       /* tensor_b */) const {
 
            static constexpr bool condition = is_type_compatible<typename AEngine::value_type, typename base_type::a_value_type>() and
                                              is_type_compatible<typename BEngine::value_type, typename base_type::b_value_type>();

            static_assert(condition, "Incorrect types for input tensors."
                                     "Ensure input types for A and B match the types indicated "
                                     "in the Precision<...> operator.");
        }

        // Fallbacks for static_asserts, only pointers or only LDB
        template<class TA, class TB, class LDB = cute::Int<base_type::this_blas_trsm_ldb>>
        CUBLASDX_DEVICE 
        cute::enable_if_t<not is_type_compatible<TA, typename base_type::a_value_type>() or
                          not is_type_compatible<TB, typename base_type::b_value_type>()>
        execute(TA const* matrix_a, TB * matrix_b, LDB ldb = {}) const {
            static constexpr bool condition = is_type_compatible<TA, typename base_type::a_value_type>() and
                                              is_type_compatible<TB, typename base_type::b_value_type>();

            static_assert(condition, "Incorrect types for input matrices."
                                     "Ensure input types for A and B match the types indicated "
                                     "in the Precision<...> operator.");
        }

        // Fallbacks for static_asserts, only LDA or (LDA and LDB)
        template<class TA, class LDA, class TB, class LDB = cute::Int<base_type::this_blas_trsm_ldb>>
        CUBLASDX_DEVICE 
        cute::enable_if_t<not is_type_compatible<TA, typename base_type::a_value_type>() or
                          not is_type_compatible<TB, typename base_type::b_value_type>()>
        execute(TA const* matrix_a, LDA lda, TB * matrix_b, LDB ldb = {}) const {
            static constexpr bool condition = is_type_compatible<TA, typename base_type::a_value_type>() and
                                              is_type_compatible<TB, typename base_type::b_value_type>();

            static_assert(condition, "Incorrect types for input matrices."
                                     "Ensure input types for A and B match the types indicated "
                                     "in the Precision<...> operator.");
        }
    };

} // namespace cublasdx::detail 
#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_BLOCK_TRSM_EXECUTION_HPP
