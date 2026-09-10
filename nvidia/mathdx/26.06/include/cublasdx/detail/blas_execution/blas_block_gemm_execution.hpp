// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_BLOCK_GEMM_EXECUTION_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_BLOCK_GEMM_EXECUTION_HPP

#include "cublasdx/detail/blas_execution/blas_execution_base.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_fwd.hpp"

namespace cublasdx::detail {

template<class... Operators>
class blas_block_gemm_execution : public blas_execution_base<blas_block_gemm_execution<Operators...>, Operators...> {
    using this_type = blas_block_gemm_execution<Operators...>;
    using base_type = blas_execution_base<this_type, Operators...>;

    // Imported shared types
    using typename base_type::this_blas_value_type;
    using typename base_type::this_blas_precision;
    using typename base_type::this_blas_size;

    public:

    // Number of elements in A, B, C matrices (includes padding / leading dimensions)
    // (ld * cols)
    static constexpr unsigned int a_size = base_type::this_blas_a_size;
    static constexpr unsigned int b_size = base_type::this_blas_b_size;
    static constexpr unsigned int c_size = base_type::this_blas_c_size;

    static constexpr auto a_shape = cute::make_shape(cute::Int<base_type::this_blas_size_m_v> {},
                                                        cute::Int<base_type::this_blas_size_k_v> {});
    using a_shape_t               = decltype(a_shape);
    static constexpr auto b_shape = cute::make_shape(cute::Int<base_type::this_blas_size_k_v> {},
                                                        cute::Int<base_type::this_blas_size_n_v> {});
    using b_shape_t               = decltype(b_shape);

    static constexpr auto c_shape = cute::make_shape(cute::Int<base_type::this_blas_size_m_v> {},
                                                     cute::Int<base_type::this_blas_size_n_v> {});
    using c_shape_t               = decltype(c_shape);

    using typename base_type::a_value_type;
    using typename base_type::b_value_type;
    using typename base_type::c_value_type;

    // Imported shared constants
    using base_type::a_alignment;
    using base_type::b_alignment;
    using base_type::c_alignment;

    static constexpr int lda = base_type::this_blas_lda;
    static constexpr int ldb = base_type::this_blas_ldb;
    static constexpr int ldc = base_type::this_blas_ldc;

    private:
    template<unsigned, result_storage, copy_kind, class, bool, int, int, class, class, class, class, class, typename, class, typename>
    friend struct rmem_unified_pipeline;

    template<unsigned, result_storage, copy_kind, class, bool, int, int, class, class, class, class, class, typename, class, typename>
    friend struct rmem_specialized_pipeline;

    template<unsigned, result_storage, copy_kind, class, bool, int, class, class, class, class, class, typename, class, typename>
    friend struct tmem_specialized_pipeline;

    template<unsigned, result_storage, copy_kind, class, bool, int, class, class, class, class, class, typename, class, typename>
    friend struct tmem_unified_pipeline;

    using base_type::has_ld;

    using this_cutlass_value_type = cutlass_value_type<this_blas_value_type>;

    /// ---- Suggestions

    using overloaded_tile_operator = typename base_type::this_blas_overloaded_tile;

    using execution_suggestion_db_t =
        cute_backend::get_database_threads<typename this_cutlass_value_type::a_type,
                                           typename this_cutlass_value_type::b_type,
                                           typename this_cutlass_value_type::c_type,
                                           this_blas_size::m,
                                           this_blas_size::n,
                                           this_blas_size::k,
                                           typename base_type::this_blas_sm,
                                           base_type::this_blas_sm_modifier_v,
                                           void,
                                           overloaded_tile_operator>;

    // Block Dimension
    // * Default value: selected by implementation
    using default_gemm_threads =
        cute::conditional_t<base_type::is_complete, execution_suggestion_db_t, cute::Int<128>>;

    using this_blas_block_dim =
        get_or_default_t<operator_type::block_dim, this_type, BlockDim<default_gemm_threads::value>>;
    static constexpr dim3 this_blas_block_dim_v = this_blas_block_dim::value;

    static constexpr int sm_v = base_type::this_blas_sm_v;

    /// ---- Backend

    // CuTe backend implementation
    using blas_backend = cute_backend::execution<typename this_cutlass_value_type::a_type,
                                                 typename this_cutlass_value_type::b_type,
                                                 typename this_cutlass_value_type::c_type,
                                                 typename this_blas_value_type::a_type,
                                                 typename this_blas_value_type::b_type,
                                                 typename this_blas_value_type::c_type,
                                                 typename base_type::this_blas_alignment,
                                                 this_blas_size::m,
                                                 this_blas_size::n,
                                                 this_blas_size::k,
                                                 typename base_type::this_blas_arrangement,
                                                 typename base_type::this_blas_transpose_mode,
                                                 typename base_type::this_blas_sm,
                                                 base_type::this_blas_sm_modifier_v,
                                                 typename base_type::this_blas_static_block_dim,
                                                 typename base_type::this_blas_streaming,
                                                 typename base_type::this_blas_with_pipeline,
                                                 this_blas_block_dim,
                                                 overloaded_tile_operator>;

    // Bugs handlers --> MACROS
    NVBUG_5668269_HANDLER;
    NVBUG_5430354_HANDLER;
    NVBUG_5697031_HANDLER;
    NVBUG_6283425_HANDLER;
    NVBUG_6283118_HANDLER;

    /// custom enable_if 
    template<typename TA, typename TB>
    static constexpr bool are_types_compatible() {
        return detail::are_types_compatible_impl<COMMONDX_STL_NAMESPACE::tuple<a_value_type, TA>,
                                                 COMMONDX_STL_NAMESPACE::tuple<b_value_type, TB>>();
    }

    template<typename TA, typename TB, typename TC>
    static constexpr bool are_types_compatible() {
        return detail::are_types_compatible_impl<COMMONDX_STL_NAMESPACE::tuple<a_value_type, TA>,
                                                 COMMONDX_STL_NAMESPACE::tuple<b_value_type, TB>,
                                                 COMMONDX_STL_NAMESPACE::tuple<c_value_type, TC>>();
    }

    template<typename Alpha, typename TA, typename TB, typename Beta, typename TC>
    static constexpr bool are_types_compatible() {
        return detail::are_types_compatible_impl<COMMONDX_STL_NAMESPACE::tuple<c_value_type, Alpha>,
                                                 COMMONDX_STL_NAMESPACE::tuple<a_value_type, TA>,
                                                 COMMONDX_STL_NAMESPACE::tuple<b_value_type, TB>,
                                                 COMMONDX_STL_NAMESPACE::tuple<c_value_type, Beta>,
                                                 COMMONDX_STL_NAMESPACE::tuple<c_value_type, TC>>();
    }

    template<typename... Ts>
    using execute_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<
        (are_types_compatible<cute::decay_t<Ts>...>())>;

    template<typename... Ts>
    using execute_disable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<
        not(are_types_compatible<cute::decay_t<Ts>...>())>;
    /// custom enable_if end

    template<matrix M>
    CUBLASDX_HOST_DEVICE constexpr static auto suggest_cute_layout_smem() {
        // TRSM uses contiguous layouts; the GEMM swizzle database does not apply.
        static_assert(not base_type::is_trsm, "Suggested layouts are not yet available for TRSM");
        // If an overload is used, turn off suggested mechanism
        if constexpr (base_type::has_overloaded_tile or base_type::has_ld) {
            return get_cute_layout<M, commondx::detail::smem_tag>();
        } else {
            using a_type = typename this_blas_value_type::a_type;
            using b_type = typename this_blas_value_type::b_type;
            using c_type = typename this_blas_value_type::c_type;

            using db_a_type = typename this_cutlass_value_type::a_type;
            using db_b_type = typename this_cutlass_value_type::b_type;
            using db_c_type = typename this_cutlass_value_type::c_type;

            constexpr bool is_a_left = base_type::this_blas_arrangement_a == arrangement::col_major;
            constexpr bool is_b_left = base_type::this_blas_arrangement_b == arrangement::col_major;
            constexpr bool is_c_left = base_type::this_blas_arrangement_c == arrangement::col_major;

            using swizzle_config = cublasdx::detail::layout_database::optimal_config_entry<
                M,
                max_threads_per_block,
                base_type::this_blas_sm_v,
                base_type::this_blas_sm_modifier_v,
                typename base_type::this_blas_streaming,
                typename base_type::this_blas_with_pipeline,
                db_a_type,
                is_a_left,
                a_alignment,
                db_b_type,
                is_b_left,
                b_alignment,
                db_c_type,
                is_c_left,
                c_alignment,
                this_blas_size::m,
                this_blas_size::n,
                this_blas_size::k>;
            if constexpr (swizzle_config::is_valid()) {
                return swizzle_config::get();
            } else {
                return get_cute_layout<M, commondx::detail::smem_tag>();
            }
        }

        CUTE_GCC_UNREACHABLE;
    }

    template<matrix M>
    CUBLASDX_HOST_DEVICE constexpr static auto tag_suggested_cute_layout_smem() {
        return commondx::detail::pointer_layout {commondx::detail::smem_tag {}, suggest_cute_layout_smem<M>()};
    }

    template<matrix M, class MemTag>
    static constexpr int get_default_ld() {
        static_assert(cute::is_same_v<MemTag, commondx::detail::smem_tag> or
                      cute::is_same_v<MemTag, commondx::detail::gmem_tag>);
        int ret = 0;
        if constexpr (cute::is_same_v<MemTag, commondx::detail::smem_tag>) {
            ret = cute::get<M>(
                cute::make_tuple(base_type::this_blas_lda, base_type::this_blas_ldb, base_type::this_blas_ldc));
        } else {
            constexpr arrangement arr = cute::get<M>(cute::make_tuple(base_type::this_blas_arrangement_a,
                                                                      base_type::this_blas_arrangement_b,
                                                                      base_type::this_blas_arrangement_c));
            ret                       = (arr == col_major)
                                            ? cute::get<M>(cute::make_tuple(this_blas_size::m, this_blas_size::k, this_blas_size::m))
                                            : cute::get<M>(cute::make_tuple(this_blas_size::k, this_blas_size::n, this_blas_size::n));
        }
        return ret;
    }

    template<matrix M, class MemTag, class LD = cute::Int<get_default_ld<M, MemTag>()>>
    CUBLASDX_HOST_DEVICE constexpr static auto get_cute_layout(LD ld = {}) {
        static_assert(cute::is_integral<LD>::value);

        using a_type = typename this_blas_value_type::a_type;
        using b_type = typename this_blas_value_type::b_type;
        using c_type = typename this_blas_value_type::c_type;

        using rows    = cute::Int<M == matrix::B ? this_blas_size::k : this_blas_size::m>;
        using columns = cute::Int<M == matrix::A ? this_blas_size::k : this_blas_size::n>;
        constexpr arrangement arr =
            cute::get<M>(cute::make_tuple(base_type::this_blas_arrangement_a,
                                            base_type::this_blas_arrangement_b,
                                            base_type::this_blas_arrangement_c));

        if constexpr (cute::is_static_v<LD>) {
            constexpr bool valid_ld =
                (ld >= ((arr == arrangement::col_major) ? rows::value : columns::value));

            static_assert(
                valid_ld || (M != matrix::A),
                "Incorrect leading dimension for A matrix, LDA must be greater than or equal to its leading size");
            static_assert(
                valid_ld || (M != matrix::B),
                "Incorrect leading dimension for B matrix, LDB must be greater than or equal to its leading size");
            static_assert(
                valid_ld || (M != matrix::C),
                "Incorrect leading dimension for C matrix, LDC must be greater than or equal to its leading size");
        }

        return cute_backend::make_layout_from_arrangement<arr>(rows {}, columns {}, ld);
    }

    // Add internal call to be able to static_assert easier on user facing calls
    template<bool GatingTurnedOff = false,
             class AEngine,
             class ALayout,
             class BEngine,
             class BLayout,
             class BlasAccumulator,
             class ALoadOp = identity,
             class BLoadOp = identity>
    CUBLASDX_DEVICE auto execute_internal(const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                                          const cublasdx::tensor<BEngine, BLayout>& tensor_b,
                                          BlasAccumulator&                          accumulator,
                                          const ALoadOp&                            a_load_op = {},
                                          const BLoadOp&                            b_load_op = {})
        -> COMMONDX_STL_NAMESPACE::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value,
                                               execute_enable_if_t<res_t<ALoadOp, typename AEngine::value_type>,
                                                                   res_t<BLoadOp, typename BEngine::value_type>,
                                                                   typename BlasAccumulator::value_type>> {
        constexpr bool ab_load_ops_are_identity = is_identity_op_v<ALoadOp> and is_identity_op_v<BLoadOp>;
        static_assert(not base_type::this_blas_streaming_v or ab_load_ops_are_identity,
                      "When EnableInputStreaming is used, A and B load ops can't be used");

        using AShape = decltype(shape(ALayout {}));
        using BShape = decltype(shape(BLayout {}));

        using a_engine_t = typename AEngine::value_type;
        using b_engine_t = typename BEngine::value_type;
        using a_result_t = res_t<ALoadOp, a_engine_t>;
        using b_result_t = res_t<BLoadOp, b_engine_t>;

        using cutlass_a_engine_t  = convert_to_cutlass_type_t<a_engine_t>;
        using cutlass_a_compute_t = convert_to_cutlass_type_t<typename this_blas_value_type::a_type>;

        using cutlass_b_engine_t  = convert_to_cutlass_type_t<b_engine_t>;
        using cutlass_b_compute_t = convert_to_cutlass_type_t<typename this_blas_value_type::b_type>;

        // Check if sizes are static
        static_assert(cute::is_static_v<AShape> && cute::is_static_v<BShape>,
                      "All layout shapes must be static, only strides can be dynamic");

        // Check if layout shapes are 2D and non-hierarchical
        // Check if layout shapes are compatible with operator defined shapes
        static_assert(rank(AShape {}) == 2 && size(cute::get<0>(AShape {})) == this_blas_size::m &&
                          size(cute::get<1>(AShape {})) == this_blas_size::k && rank(BShape {}) == 2 &&
                          size(cute::get<0>(BShape {})) == this_blas_size::k &&
                          size(cute::get<1>(BShape {})) == this_blas_size::n,
                      "Tensor API currently supports only \
             hierarchical 2D tensors sizes of which \
             match operator provided sizes");

        // Input types check
        static_assert(
            (sizeof(a_engine_t) == sizeof(a_value_type) and alignof(a_engine_t) == alignof(a_value_type) and
             sizeof(b_engine_t) == sizeof(b_value_type) and alignof(b_engine_t) == alignof(b_value_type)) or
                base_type::has_alignment,
            "If using data types decoupled from computation precision, Alignment operator must be set");

        static_assert(not base_type::this_blas_streaming::value or
                          cute::is_same_v<cutlass_a_engine_t, cutlass_a_compute_t>,
                      "When EnableInputStreaming is used decoupled precision is not available");

        static_assert(not base_type::this_blas_streaming::value or
                          cute::is_same_v<cutlass_b_engine_t, cutlass_b_compute_t>,
                      "When EnableInputStreaming is used decoupled precision is not available");

        // Alignment checks
        static_assert(((base_type::this_blas_alignment_a % alignof(a_engine_t)) == 0) &&
                          (base_type::this_blas_alignment_a >= alignof(a_engine_t)),
                      "Incorrect alignment for matrix A; it has to be a multiple of type of matrix A");
        static_assert(((base_type::this_blas_alignment_b % alignof(b_engine_t)) == 0) &&
                          (base_type::this_blas_alignment_b >= alignof(b_engine_t)),
                      "Incorrect alignment for matrix B; it has to be a multiple of type of matrix B");

        // Functor checks
        static_assert(
            is_functor_compatible<ALoadOp, a_engine_t, a_value_type>(),
            "ALoadOp functor must accept value of tensor_a type and return value convertible to tensor_a type");
        static_assert(
            is_functor_compatible<BLoadOp, b_engine_t, b_value_type>(),
            "BLoadOp functor must accept value of tensor_b type and return value convertible to tensor_b type");

        static_assert(cute::is_static<ALoadOp>::value, "ALoadOp functor must be stateless");
        static_assert(cute::is_static<BLoadOp>::value, "BLoadOp functor must be stateless");

        auto cutlass_a_load_op = transform_op_wrapper<ALoadOp, a_engine_t> {};
        auto cutlass_b_load_op = transform_op_wrapper<BLoadOp, b_engine_t> {};

        blas_backend::template tensor_gemm<GatingTurnedOff>(safe_recast<cutlass_a_engine_t>(tensor_a),
                                                       safe_recast<cutlass_b_engine_t>(tensor_b),
                                                       accumulator,
                                                       cutlass_a_load_op,
                                                       cutlass_b_load_op);
    }

    // Allow only WithPipeline() to call into execute() accepting tile_pipeline 
    template<class TilePipeline, class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
    CUBLASDX_DEVICE auto execute_from_pipeline(TilePipeline&     tile_pipeline,
                                               BlasAccumulator&& accumulator,
                                               const ALoadOp&    a_load_op = {},
                                               const BLoadOp&    b_load_op = {})
        -> COMMONDX_STL_NAMESPACE::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value and
                                               is_pipeline<TilePipeline>::value> {
        if (!tile_pipeline.should_issue_mma()) return;

        tile_pipeline.compute_smem_acquire();
        auto [smem_a, smem_b] = tile_pipeline.compute_smem_tensors();
        execute_internal<true>(smem_a, smem_b, accumulator, a_load_op, b_load_op);
        tile_pipeline.compute_smem_commit();
    }

    public:

    template<matrix M, class MemTag, class LD = cute::Int<get_default_ld<M, MemTag>()>>
    CUBLASDX_HOST_DEVICE constexpr static auto tag_cute_layout(LD ld = {}) {
        return commondx::detail::pointer_layout {MemTag {}, get_cute_layout<M, MemTag>(ld)};
    }

    static constexpr instruction_type swizzled_instruction_kind = blas_backend::swizzled_instruction_kind;
    static constexpr instruction_type default_instruction_kind  = blas_backend::default_instruction_kind;

    static constexpr CUBLASDX_HOST_DEVICE dim3 get_suggested_block_dim() {
        static_assert(base_type::is_complete,
                      "Can't provide suggested block dimensions, description is not complete");
        return default_gemm_threads::value;
    }

    static constexpr CUBLASDX_HOST_DEVICE dim3 get_block_dim() {
        static_assert(base_type::is_complete, "Can't provide block dimensions, description is not complete");
        if constexpr (base_type::has_block_dim) {
            return this_blas_block_dim_v;
        }
        return get_suggested_block_dim();
    }

    static constexpr dim3 suggested_block_dim = this_type::get_suggested_block_dim();
    static constexpr dim3 block_dim = this_type::get_block_dim();
    static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;

    template<typename... Ts>
    CUBLASDX_HOST_DEVICE constexpr static auto suggest_layout_smem_a(Ts... ts) {
        return tag_suggested_cute_layout_smem<matrix::A>(ts...);
    }

    template<typename... Ts>
    CUBLASDX_HOST_DEVICE constexpr static auto suggest_layout_smem_b(Ts... ts) {
        return tag_suggested_cute_layout_smem<matrix::B>(ts...);
    }

    template<typename... Ts>
    CUBLASDX_HOST_DEVICE constexpr static auto suggest_layout_smem_c(Ts... ts) {
        return tag_suggested_cute_layout_smem<matrix::C>(ts...);
    }

    // ---- GEMM executes ----------------------------------------------

    template<class AEngine,
             class ALayout,
             class BEngine,
             class BLayout,
             class BlasAccumulator,
             class ALoadOp = identity,
             class BLoadOp = identity>
    CUBLASDX_DEVICE auto execute(const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                                 const cublasdx::tensor<BEngine, BLayout>& tensor_b,
                                 BlasAccumulator&                          accumulator,
                                 const ALoadOp&                            a_load_op = {},
                                 const BLoadOp&                            b_load_op = {})
        -> COMMONDX_STL_NAMESPACE::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value,
                                               execute_enable_if_t<res_t<ALoadOp, typename AEngine::value_type>,
                                                                   res_t<BLoadOp, typename BEngine::value_type>,
                                                                   typename BlasAccumulator::value_type>> {
        execute_internal(tensor_a, tensor_b, accumulator, a_load_op, b_load_op);
    }

    template<class AEngine,
             class ALayout,
             class BEngine,
             class BLayout,
             class BlasAccumulator,
             class ALoadOp = identity,
             class BLoadOp = identity>
    CUBLASDX_DEVICE auto execute(const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                                 const cublasdx::tensor<BEngine, BLayout>& tensor_b,
                                 BlasAccumulator&&                         accumulator,
                                 const ALoadOp&                            a_load_op = {},
                                 const BLoadOp&                            b_load_op = {})
        -> COMMONDX_STL_NAMESPACE::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value,
                                               execute_enable_if_t<res_t<ALoadOp, typename AEngine::value_type>,
                                                                   res_t<BLoadOp, typename BEngine::value_type>,
                                                                   typename BlasAccumulator::value_type>> {
        execute_internal(tensor_a, tensor_b, accumulator, a_load_op, b_load_op);
    }

    template<class TilePipeline, class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
    CUBLASDX_DEVICE auto execute(TilePipeline&     tile_pipeline,
                                 BlasAccumulator&& accumulator,
                                 const ALoadOp&    a_load_op = {},
                                 const BLoadOp&    b_load_op = {})
        -> COMMONDX_STL_NAMESPACE::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value and
                                               is_tile_pipeline<TilePipeline>::value> {
        tile_pipeline.compute_smem_acquire();
        auto [smem_a, smem_b] = tile_pipeline.compute_smem_tensors();
        execute_internal(smem_a, smem_b, accumulator, a_load_op, b_load_op);
        tile_pipeline.compute_smem_commit();
    }

    template<class TilePipeline,
             class ALoadOp                                                               = identity,
             class BLoadOp                                                               = identity,
             COMMONDX_STL_NAMESPACE::enable_if_t<is_tile_pipeline<TilePipeline>::value>* = nullptr>
    CUBLASDX_DEVICE auto execute(TilePipeline&  tile_pipeline,
                                 const ALoadOp& a_load_op = {},
                                 const BLoadOp& b_load_op = {}) {
        tile_pipeline.compute_smem_acquire();
        auto [smem_a, smem_b] = tile_pipeline.compute_smem_tensors();
        auto accumulator      = execute(smem_a, smem_b, a_load_op, b_load_op);
        tile_pipeline.compute_smem_commit();
        return accumulator;
    }

    template<class AEngine,
             class ALayout,
             class BEngine,
             class BLayout,
             class ALoadOp                                                      = identity,
             class BLoadOp                                                      = identity,
             execute_enable_if_t<res_t<ALoadOp, typename AEngine::value_type>,
                                 res_t<BLoadOp, typename BEngine::value_type>>* = nullptr>
    CUBLASDX_DEVICE auto execute(const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                                 const cublasdx::tensor<BEngine, BLayout>& tensor_b,
                                 const ALoadOp&                            a_load_op = {},
                                 const BLoadOp&                            b_load_op = {}) {

        using a_engine_t         = typename AEngine::value_type;
        using b_engine_t         = typename BEngine::value_type;
        using cutlass_a_engine_t = convert_to_cutlass_type_t<a_engine_t>;
        using cutlass_b_engine_t = convert_to_cutlass_type_t<b_engine_t>;

        // Choose between default and swizzled accumulator
        auto accumulator = blas_backend::choose_accumulator(safe_recast<cutlass_a_engine_t>(tensor_a),
                                                                safe_recast<cutlass_b_engine_t>(tensor_b));

        // Call GEMM
        execute(tensor_a, tensor_b, accumulator, a_load_op, b_load_op);

        // Return rich-type accumulator
        return accumulator;
    }

    // C in Shared Memory API
    template<class Alpha,
             class AEngine,
             class ALayout,
             class BEngine,
             class BLayout,
             class Beta,
             class CEngine,
             class CLayout,
             class ALoadOp  = identity,
             class BLoadOp  = identity,
             class CLoadOp  = identity,
             class CStoreOp = identity>
    CUBLASDX_DEVICE auto execute(const Alpha&                              alpha,
                                 const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                                 const cublasdx::tensor<BEngine, BLayout>& tensor_b,
                                 const Beta&                               beta,
                                 cublasdx::tensor<CEngine, CLayout>&       tensor_c,
                                 const ALoadOp&                            a_load_op  = {},
                                 const BLoadOp&                            b_load_op  = {},
                                 const CLoadOp&                            c_load_op  = {},
                                 const CStoreOp&                           c_store_op = {})
        -> execute_enable_if_t<Alpha, // Alpha
                               res_t<ALoadOp, typename AEngine::value_type>,
                               res_t<BLoadOp, typename BEngine::value_type>,
                               Beta, // Beta
                               res_t<CLoadOp, typename CEngine::value_type>> {

        using CShape     = decltype(shape(CLayout {}));
        using c_engine_t = typename CEngine::value_type;

        // All checks are performed only for C matrix, as A and B are already checked in register execute
        // ==============================================================================================
        // Check if sizes are static
        static_assert(cute::is_static_v<CShape>,
                      "All layout shapes must be static, only strides can be dynamic");

        // Check if layout shapes are 2D and non-hierarchical
        // Check if layout shapes are compatible with operator defined shapes
        static_assert(rank(CShape {}) == 2 && size(cute::get<0>(CShape {})) == this_blas_size::m &&
                          size(cute::get<1>(CShape {})) == this_blas_size::n,
                      "Tensor API currently supports only hierarchical 2D tensors sizes of which match "
                      "operator provided sizes");

        static_assert(
            (sizeof(c_engine_t) == sizeof(c_value_type) and alignof(c_engine_t) == alignof(c_value_type)) or
                base_type::has_alignment,
            "If using data types decoupled from computation precision, Alignment operator must be set");

        // Alignment check for C
        static_assert(((base_type::this_blas_alignment_c % alignof(c_engine_t)) == 0) &&
                          (base_type::this_blas_alignment_c >= alignof(c_engine_t)),
                      "Incorrect alignment for matrix C; it has to be a multiple of type of matrix C");
        // ==============================================================================================

        // C Functor checks (A and B take place in register execute)
        static_assert(is_functor_compatible<CLoadOp, c_engine_t, c_value_type>(),
                      "CLoadOp functor must accept value of tensor_c type and return value convertible to "
                      "BLAS::c_value_type");
        static_assert(is_functor_compatible<CStoreOp, c_value_type, c_engine_t>(),
                      "CStoreOp functor must accept value of tensor_c type and return value convertible to "
                      "BLAS::c_value_type");

        // C = A * B; GEMM with output in registers
        auto accumulator = execute(tensor_a, tensor_b, a_load_op, b_load_op);
        if (accumulator.is_thread_active()) {
            accumulator.axpby(alpha, beta, tensor_c, c_load_op, c_store_op);
        }
    }

    // C in Shared Memory API
    template<class Alpha,
             class AEngine,
             class ALayout,
             class BEngine,
             class BLayout,
             class Beta,
             class CEngine,
             class CLayout,
             class ALoadOp  = identity,
             class BLoadOp  = identity,
             class CLoadOp  = identity,
             class CStoreOp = identity>
    CUBLASDX_DEVICE auto execute(const Alpha&                              alpha,
                                 const cublasdx::tensor<AEngine, ALayout>& tensor_a,
                                 const cublasdx::tensor<BEngine, BLayout>& tensor_b,
                                 const Beta&                               beta,
                                 cublasdx::tensor<CEngine, CLayout>&&      tensor_c,
                                 const ALoadOp&                            a_load_op  = {},
                                 const BLoadOp&                            b_load_op  = {},
                                 const CLoadOp&                            c_load_op  = {},
                                 const CStoreOp&                           c_store_op = {})
        -> execute_enable_if_t<Alpha, // Alpha
                               res_t<ALoadOp, typename AEngine::value_type>,
                               res_t<BLoadOp, typename BEngine::value_type>,
                               Beta, // Beta
                               res_t<CLoadOp, typename CEngine::value_type>> {
        execute(alpha, tensor_a, tensor_b, beta, tensor_c, a_load_op, b_load_op, c_load_op, c_store_op);
    }

    template<class Alpha,
             class AEngine,
             class ALayout,
             class BEngine,
             class BLayout,
             class Beta,
             class CEngine,
             class CLayout,
             class ALoadOp,
             class BLoadOp,
             class CLoadOp,
             class CStoreOp>
    CUBLASDX_DEVICE auto execute(const Alpha& /* alpha */,
                                 const cublasdx::tensor<AEngine, ALayout>& /* tensor_a */,
                                 const cublasdx::tensor<BEngine, BLayout>& /* tensor_b */,
                                 const Beta& /* beta */,
                                 cublasdx::tensor<CEngine, CLayout>& /* tensor_c */,
                                 [[maybe_unused]] const ALoadOp&  a_load_op  = identity {},
                                 [[maybe_unused]] const BLoadOp&  b_load_op  = identity {},
                                 [[maybe_unused]] const CLoadOp&  c_load_op  = identity {},
                                 [[maybe_unused]] const CStoreOp& c_store_op = identity {})
        -> execute_disable_if_t<Alpha, // Must be compatible (size, alignment) with c_value_type
                                res_t<ALoadOp, typename AEngine::value_type>,
                                res_t<BLoadOp, typename BEngine::value_type>,
                                Beta, // Must be compatible (size, alignment) with c_value_type
                                    res_t<CLoadOp, typename CEngine::value_type>> {

        constexpr bool condition = are_types_compatible<Alpha,
                                                        res_t<ALoadOp, typename AEngine::value_type>,
                                                        res_t<BLoadOp, typename BEngine::value_type>,
                                                        Beta,
                                                        res_t<CLoadOp, typename CEngine::value_type>>();

        static_assert(condition, "Incorrect types for inputs, LD operator used or TransposeMode used. \
            Ensure input types for A, B and C match the types \
            indicated in the Precision<...> operator.");
    }

    // T - can be any type if its alignment and size are the same as those of ::value_type
    template<class Alpha,
             class TA,
             class LDA,
             class TB,
             class LDB,
             class Beta,
             class TC,
             class LDC,
             class ALoadOp  = identity,
             class BLoadOp  = identity,
             class CLoadOp  = identity,
             class CStoreOp = identity>
    CUBLASDX_DEVICE auto execute(const Alpha&    alpha,
                                 TA*             matrix_a,
                                 const LDA       lda,
                                 TB*             matrix_b,
                                 const LDB       ldb,
                                 const Beta&     beta,
                                 TC*             matrix_c,
                                 const LDC       ldc,
                                 const ALoadOp&  a_load_op  = {},
                                 const BLoadOp&  b_load_op  = {},
                                 const CLoadOp&  c_load_op  = {},
                                 const CStoreOp& c_store_op = {}) //
        -> execute_enable_if_t<Alpha, TA, TB, Beta, TC> {
        static_assert(
            cute::is_integral<LDA>::value and cute::is_integral<LDB>::value and cute::is_integral<LDC>::value,
            "LD values must be either static or dynamic integral types");
        cute::Tensor ta = cublasdx::make_tensor(
            cute::make_smem_ptr(matrix_a),
            cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_a>(
                cute::Int<this_blas_size::m> {}, cute::Int<this_blas_size::k> {}, lda));
        cute::Tensor tb = cublasdx::make_tensor(
            cute::make_smem_ptr(matrix_b),
            cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_b>(
                cute::Int<this_blas_size::k> {}, cute::Int<this_blas_size::n> {}, ldb));
        cute::Tensor tc = cublasdx::make_tensor(
            cute::make_smem_ptr(matrix_c),
            cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_c>(
                cute::Int<this_blas_size::m> {}, cute::Int<this_blas_size::n> {}, ldc));

        execute(
            alpha,
            ta,
            tb,
            beta,
            tc,
            compose_functors(cute_backend::get_load_op_from_transpose<base_type::this_blas_transpose_mode_a>(),
                             a_load_op),
            compose_functors(cute_backend::get_load_op_from_transpose<base_type::this_blas_transpose_mode_b>(),
                             b_load_op),
            c_load_op,
            c_store_op);
    }

    template<class Alpha,
             class TA,
             class TB,
             class Beta,
             class TC,
             class ALoadOp  = identity,
             class BLoadOp  = identity,
             class CLoadOp  = identity,
             class CStoreOp = identity>
    CUBLASDX_DEVICE auto execute(const Alpha     alpha,
                                 TA*             matrix_a,
                                 TB*             matrix_b,
                                 const Beta      beta,
                                 TC*             matrix_c,
                                 const ALoadOp&  a_load_op  = {},
                                 const BLoadOp&  b_load_op  = {},
                                 const CLoadOp&  c_load_op  = {},
                                 const CStoreOp& c_store_op = {}) //
        -> execute_enable_if_t<Alpha, TA, TB, Beta, TC> {

        auto lda = cute::Int<base_type::this_blas_lda> {};
        auto ldb = cute::Int<base_type::this_blas_ldb> {};
        auto ldc = cute::Int<base_type::this_blas_ldc> {};

        execute(alpha,
                matrix_a,
                lda,
                matrix_b,
                ldb,
                beta,
                matrix_c,
                ldc,
                a_load_op,
                b_load_op,
                c_load_op,
                c_store_op);
    }

    template<class Alpha,
             class TA,
             class TB,
             class Beta,
             class TC,
             class ALoadOp  = identity,
             class BLoadOp  = identity,
             class CLoadOp  = identity,
             class CStoreOp = identity>
    CUBLASDX_DEVICE auto execute(const Alpha /* alpha */,
                                 TA* /* matrix_a */,
                                 const unsigned int /* lda */,
                                 TB* /* matrix_b */,
                                 const unsigned int /* ldb */,
                                 const Beta /* beta */,
                                 TC* /* matrix_c */,
                                 const unsigned int /* ldc */,
                                 const ALoadOp& /* a_load_op */   = {},
                                 const BLoadOp& /* b_load_op */   = {},
                                 const CLoadOp& /* c_load_op */   = {},
                                 const CStoreOp& /* c_store_op */ = {}) //
        -> execute_disable_if_t<Alpha, TA, TB, Beta, TC> {
        static constexpr bool condition = are_types_compatible<Alpha, TA, TB, Beta, TC>();

        static_assert(condition, "Incorrect types for inputs or lacking TransposeMode operator.  \
        Ensure input types for A, B and C match the types \
        indicated in the Precision<...> operator.");
    }

    template<class Alpha,
             class TA,
             class TB,
             class Beta,
             class TC,
             class ALoadOp  = identity,
             class BLoadOp  = identity,
             class CLoadOp  = identity,
             class CStoreOp = identity>
    CUBLASDX_DEVICE auto execute(const Alpha /* alpha */,
                                 TA* /* matrix_a */,
                                 TB* /* matrix_b */,
                                 const Beta /* beta */,
                                 TC* /* matrix_c */,
                                 const ALoadOp& /* a_load_op */   = {},
                                 const BLoadOp& /* b_load_op */   = {},
                                 const CLoadOp& /* c_load_op */   = {},
                                 const CStoreOp& /* c_store_op */ = {}) //
        -> execute_disable_if_t<Alpha, TA, TB, Beta, TC> {
        static constexpr bool condition = are_types_compatible<Alpha, TA, TB, Beta, TC>();

        static_assert(condition, "Incorrect types for inputs or lacking TransposeMode operator.  \
        Ensure input types for A, B and C match the types \
        indicated in the Precision<...> operator.");
    }

    using default_accumulator_t = typename blas_backend::default_accumulator_t;
    using suggested_accumulator_t = cute::conditional_t<base_type::has_overloaded_tile or base_type::has_ld,
                                                       default_accumulator_t,
                                                       typename blas_backend::suggested_accumulator_t>;

    static CUBLASDX_DEVICE default_accumulator_t get_accumulator_impl() { return blas_backend::get_accumulator(); }

    static CUBLASDX_DEVICE suggested_accumulator_t suggest_accumulator_impl() {
        if constexpr (base_type::has_overloaded_tile or base_type::has_ld) {
            return blas_backend::get_accumulator();
        } else {
            return blas_backend::suggest_accumulator();
        }
        CUTE_GCC_UNREACHABLE;
    }

    static CUBLASDX_DEVICE default_accumulator_t get_accumulator() {
        static_assert(!base_type::this_blas_with_pipeline_v,
                      "In pipelined mode you are required to call tile_pipeline.get_accumulator() instead");
        return get_accumulator_impl();
    }

    static CUBLASDX_DEVICE suggested_accumulator_t suggest_accumulator() {
        static_assert(!base_type::this_blas_with_pipeline_v,
                      "In pipelined mode you are required to call tile_pipeline.get_accumulator() instead");
        return suggest_accumulator_impl();
    }

};

} // namespace cublasdx::detail 
#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_BLAS_BLOCK_GEMM_EXECUTION_HPP
