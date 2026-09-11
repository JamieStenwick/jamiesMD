#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_THREAD_TRSM_EXECUTION_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_THREAD_TRSM_EXECUTION_HPP

#include "cublasdx/detail/blas_execution/blas_execution_base.hpp"

namespace cublasdx::detail {

    template<class... Operators>
    class blas_thread_trsm_execution : public blas_execution_base<blas_thread_trsm_execution<Operators...>, Operators...> {
        using this_type = blas_thread_trsm_execution<Operators...>;
        using base_type = blas_execution_base<this_type, Operators...>;

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
        using typename base_type::this_blas_size;

        template<matrix M, class MemTag>
        static constexpr int get_default_ld() {
            // TRSM smem: A is dim_axdim_a, B is MxN.
            if (M == matrix::A) {
                return static_cast<int>(base_type::this_blas_trsm_lda);
            } 
            
            return static_cast<int>(base_type::this_blas_trsm_ldb);
        }


        template<matrix M, class MemTag, class LD = cute::Int<get_default_ld<M, MemTag>()>>
        CUBLASDX_HOST_DEVICE constexpr static auto get_cute_layout(LD ld = {}) {
            static_assert(cute::is_integral<LD>::value);
            if constexpr (M == matrix::A) {
                constexpr unsigned    dim_a = base_type::this_blas_trsm_dim_a;
                constexpr arrangement arr   = base_type::this_blas_arrangement_a;
                return cute_backend::make_layout_from_arrangement<arr>(
                    cute::Int<dim_a> {}, cute::Int<dim_a> {}, ld);
            } else {
                constexpr arrangement arr = base_type::this_blas_arrangement_b;
                return cute_backend::make_layout_from_arrangement<arr>(
                    cute::Int<this_blas_size::m> {},
                    cute::Int<this_blas_size::n> {},
                    ld);
            }

            CUTE_GCC_UNREACHABLE;
        }

        public:

        template<matrix M, class MemTag, class LD = cute::Int<get_default_ld<M, MemTag>()>>
        CUBLASDX_HOST_DEVICE constexpr static auto tag_cute_layout(LD ld = {}) {
            return commondx::detail::pointer_layout {MemTag {}, get_cute_layout<M, MemTag>(ld)};
        }

        static constexpr unsigned int a_size = base_type::this_blas_trsm_a_size;
        static constexpr unsigned int b_size = base_type::this_blas_trsm_b_size;

        // ---- Thread-level TRSM execute ----------------------------------
        // Each CUDA thread independently solves one MxN triangular system.
        // Tensors may live in any address space (global, registers, etc.).

        // Tensor API
        template<class AEngine, class ALayout, class BEngine, class BLayout>
        CUBLASDX_DEVICE
            cute::enable_if_t<is_type_compatible<typename AEngine::value_type, typename base_type::a_value_type>() and
                              is_type_compatible<typename BEngine::value_type, typename base_type::b_value_type>()>
            execute(const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                    cublasdx::tensor<BEngine, BLayout>&       tensor_b) const {
            auto input_a = safe_recast<typename base_type::a_value_type>(tensor_a);
            auto input_b = safe_recast<typename base_type::b_value_type>(tensor_b);

            trsm::thread_execute<base_type::this_blas_size_m_v,
                                 base_type::this_blas_size_n_v,
                                 base_type::this_blas_side_v,
                                 base_type::this_blas_diag_v,
                                 base_type::this_blas_fill_mode_v,
                                 base_type::this_blas_alignment_b>(input_a, input_b);
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

        // Fallbacks for static_asserts
        // only pointers or only LDB
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

        // only LDA or (LDA and LDB)
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
#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_THREAD_TRSM_EXECUTION_HPP
