// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_HPP
#define CUBLASDX_DETAIL_PIPELINE_HPP

#include "cublasdx/detail/pipeline/accumulator_mode.hpp"
#include "cublasdx/traits.hpp"

#if defined(CUBLASDX_OVERLOAD_DEVICE_PIPELINE_CREATION) || !defined(COMMONDX_DETAIL_USE_CUDA_STL)
#    include "cublasdx/detail/pipeline/pipeline_result.hpp"
#    include "cublasdx/detail/pipeline/device/copy_atoms.hpp"
#    include "cublasdx/detail/pipeline/device/device_pipeline.hpp"

namespace cublasdx {

    template<class BLAS, class ValueTypeA, class ValueTypeB, class ValueTypeC, class SmemLayoutA, class SmemLayoutB, class SmemLayoutC>
    constexpr int suggest_max_pipeline_depth(SmemLayoutA const& smem_layout_a, SmemLayoutB const& smem_layout_b, SmemLayoutC const& smem_layout_c) {
        constexpr unsigned max_alignment = 128;
        constexpr unsigned stage_shared_req = cutlass::round_up(cublasdx::cosize(decltype(detail::get_cute_layout(smem_layout_a)){}) * sizeof(ValueTypeA), max_alignment) +
                                              cutlass::round_up(cublasdx::cosize(decltype(detail::get_cute_layout(smem_layout_b)){}) * sizeof(ValueTypeB), max_alignment) +
                                              sizeof(cublasdx::pipeline_stage_scratch_t);

        constexpr unsigned c_size = cublasdx::cosize(decltype(detail::get_cute_layout(smem_layout_c)){}) * sizeof(ValueTypeC);
        constexpr unsigned available_shared_memory = commondx::device_info<sm_of_v<BLAS>>::shared_memory() - c_size;
        return cute::min(16, (available_shared_memory) / stage_shared_req);
    }

    template<class BLAS, class ValueTypeA, class ValueTypeB, class SmemLayoutA, class SmemLayoutB>
    constexpr int suggest_max_pipeline_depth(SmemLayoutA const& smem_layout_a, SmemLayoutB const& smem_layout_b) {
        constexpr unsigned max_alignment = 128;
        constexpr unsigned stage_shared_req = cutlass::round_up(cublasdx::cosize(decltype(detail::get_cute_layout(smem_layout_a)){}) * sizeof(ValueTypeA), max_alignment) +
                                              cutlass::round_up(cublasdx::cosize(decltype(detail::get_cute_layout(smem_layout_b)){}) * sizeof(ValueTypeB), max_alignment) +
                                              sizeof(cublasdx::pipeline_stage_scratch_t);

        constexpr unsigned available_shared_memory = commondx::device_info<sm_of_v<BLAS>>::shared_memory();
        // Shared memory reserved for future uses / storing required additional pointers to data
        constexpr unsigned reserved_shared_memory = 32;
        return cute::min(16, (available_shared_memory - reserved_shared_memory) / stage_shared_req);
    }

    template<class BLAS, class ValueTypeA, class ValueTypeB>
    constexpr int suggest_max_pipeline_depth() {
        return suggest_max_pipeline_depth<BLAS, ValueTypeA, ValueTypeB>(BLAS::suggest_layout_smem_a(), BLAS::suggest_layout_smem_b());
    }

    namespace detail {
        enum class blocksize_strategy
        {
            fixed,
            heuristic
        };
    }

    template<class BLAS,
             class AGmemTensor,
             class BGmemTensor,
             class SmemLayoutA,
             class SmemLayoutB,
             detail::blocksize_strategy BlocksizeStrategy>
    struct pipeline_helper {
        // Type aliases for tensor and layout types
        using io_type_a       = cute::remove_cvref_t<typename AGmemTensor::value_type>;
        using io_type_b       = cute::remove_cvref_t<typename BGmemTensor::value_type>;
        using tensor_a_t      = AGmemTensor;
        using tensor_b_t      = BGmemTensor;
        using smem_layout_a_t = SmemLayoutA;
        using smem_layout_b_t = SmemLayoutB;
        using layout_a_t      = cute::remove_cvref_t<decltype(cublasdx::detail::get_cute_layout(
            COMMONDX_STL_NAMESPACE::declval<SmemLayoutA>()))>;
        using layout_b_t      = cute::remove_cvref_t<decltype(cublasdx::detail::get_cute_layout(
            COMMONDX_STL_NAMESPACE::declval<SmemLayoutB>()))>;

        // Blocksize strategy
        static constexpr bool disable_warp_specialization = BlocksizeStrategy == detail::blocksize_strategy::fixed;

        // Rank detection
        static constexpr int  a_rank = decltype(cute::rank(COMMONDX_STL_NAMESPACE::declval<tensor_a_t>()))::value;
        static constexpr int  b_rank = decltype(cute::rank(COMMONDX_STL_NAMESPACE::declval<tensor_b_t>()))::value;
        static constexpr bool is_2d  = a_rank == 2 && b_rank == 2;
        static constexpr bool is_3d  = a_rank == 3 && b_rank == 3;
        static_assert(is_2d or is_3d, "GMEM must be either 2D or 3D not mixed");

        using per_cta_a_shape = typename BLAS::a_shape_t;
        using per_cta_b_shape = typename BLAS::b_shape_t;

        // Conditional coordinate and tiler based on rank
        static constexpr auto dummy_coord =
            cute::conditional_return<is_2d>(cute::make_coord(0, 0), cute::make_coord(0, 0, 0));
        static constexpr auto a_tiler =
            cute::conditional_return<is_2d>(per_cta_a_shape {},
                                            cute::append<3>(per_cta_a_shape {}, cute::_1 {}));
        static constexpr auto b_tiler =
            cute::conditional_return<is_2d>(per_cta_b_shape {},
                                            cute::append<3>(per_cta_b_shape {}, cute::_1 {}));

        using a_tiler_t = cute::remove_cvref_t<decltype(a_tiler)>;
        using b_tiler_t = cute::remove_cvref_t<decltype(b_tiler)>;

        // Static constexpr computations for arrangement, alignment, and compatibility
        using stride_0_a = decltype(cute::stride<0>(COMMONDX_STL_NAMESPACE::declval<tensor_a_t>()));
        using stride_1_a = decltype(cute::stride<1>(COMMONDX_STL_NAMESPACE::declval<tensor_a_t>()));
        static_assert(cute::is_constant<1, stride_0_a>::value or cute::is_constant<1, stride_1_a>::value,
                "a_gmem_tensor must be either row or column major");

        using stride_0_b = decltype(cute::stride<0>(COMMONDX_STL_NAMESPACE::declval<tensor_b_t>()));
        using stride_1_b = decltype(cute::stride<1>(COMMONDX_STL_NAMESPACE::declval<tensor_b_t>()));
        static_assert(cute::is_constant<1, stride_0_b>::value or cute::is_constant<1, stride_1_b>::value,
                "b_gmem_tensor must be either row or column major");

        static constexpr arrangement smem_arrangement_a = cublasdx::arrangement_of_v_a<BLAS>;
        static constexpr arrangement smem_arrangement_b = cublasdx::arrangement_of_v_b<BLAS>;
        static constexpr arrangement gmem_arrangement_a = cute::is_constant<1, stride_0_a>::value
                                                              ? arrangement::col_major
                                                              : arrangement::row_major;
        static constexpr arrangement gmem_arrangement_b = cute::is_constant<1, stride_0_b>::value
                                                              ? arrangement::col_major
                                                              : arrangement::row_major;
        static constexpr bool is_smem_arrangement_same_as_gmem_a = smem_arrangement_a == gmem_arrangement_a;
        static constexpr bool is_smem_arrangement_same_as_gmem_b = smem_arrangement_b == gmem_arrangement_b;

        static constexpr unsigned max_alignment_layout_a = cute::max_alignment(layout_a_t {}) * sizeof(io_type_a);
        static constexpr unsigned max_alignment_layout_b = cute::max_alignment(layout_b_t {}) * sizeof(io_type_b);
        static constexpr unsigned lowest_alignment       = cute::min(cublasdx::alignment_of_v_a<BLAS>,
                                                               cublasdx::alignment_of_v_b<BLAS>,
                                                               max_alignment_layout_a,
                                                               max_alignment_layout_b);

        static constexpr bool tma_a_layout_compat =
            cublasdx::detail::is_tma_compatible_layout<io_type_a>(layout_a_t {});
        static constexpr bool tma_b_layout_compat =
            cublasdx::detail::is_tma_compatible_layout<io_type_b>(layout_b_t {});

        static constexpr int tma_alignment_requirement_bytes = 16;

        static constexpr bool tma_layout_compat = tma_a_layout_compat && tma_b_layout_compat;
        static constexpr bool tma_align_compat  = (lowest_alignment % tma_alignment_requirement_bytes) == 0;
        static constexpr int  blas_sm           = cublasdx::sm_of_v<BLAS>;
        static constexpr bool tma_sm_compat     = blas_sm >= 900;

        // Tile layout type computations using conditional tilers
        using gmem_layout_tile_a_t =
            decltype(cute::local_tile(COMMONDX_STL_NAMESPACE::declval<tensor_a_t>(), a_tiler, dummy_coord).layout());
        static constexpr int max_common_a =
            decltype(cute::max_common_vector(COMMONDX_STL_NAMESPACE::declval<gmem_layout_tile_a_t>(),
                                             layout_a_t {}))::value;
        static constexpr bool a_box_orientation = max_common_a >= (tma_alignment_requirement_bytes / sizeof(io_type_a));

        using gmem_layout_tile_b_t =
            decltype(cute::local_tile(COMMONDX_STL_NAMESPACE::declval<tensor_b_t>(), b_tiler, dummy_coord).layout());
        static constexpr int max_common_b =
            decltype(cute::max_common_vector(COMMONDX_STL_NAMESPACE::declval<gmem_layout_tile_b_t>(),
                                             layout_b_t {}))::value;
        static constexpr bool b_box_orientation = max_common_b >= (tma_alignment_requirement_bytes / sizeof(io_type_b));

        static constexpr bool is_tma_possible =
            #if defined(__clang_major__) and (__clang_major__ <= 9)
            false;
            #else
            tma_layout_compat && tma_align_compat && tma_sm_compat && a_box_orientation && b_box_orientation;
            #endif

        // Copy type computations using conditional tilers
        using coop_copy_type_a =
            decltype(cublasdx::get_copy_type<BLAS::max_threads_per_block, lowest_alignment, blas_sm>(
                COMMONDX_STL_NAMESPACE::declval<
                    decltype(cute::local_tile(COMMONDX_STL_NAMESPACE::declval<tensor_a_t>(), a_tiler, dummy_coord))>(),
                cublasdx::make_tensor(cute::make_smem_ptr<io_type_a>(nullptr), layout_a_t {})));
        using coop_copy_type_b =
            decltype(cublasdx::get_copy_type<BLAS::max_threads_per_block, lowest_alignment, blas_sm>(
                COMMONDX_STL_NAMESPACE::declval<
                    decltype(cute::local_tile(COMMONDX_STL_NAMESPACE::declval<tensor_b_t>(), b_tiler, dummy_coord))>(),
                cublasdx::make_tensor(cute::make_smem_ptr<io_type_b>(nullptr), layout_b_t {})));
        static constexpr cublasdx::detail::copy_kind coop_copy_kind_a =
            cublasdx::detail::convert_cute_to_copy_kind_v<coop_copy_type_a>;
        static constexpr cublasdx::detail::copy_kind coop_copy_kind_b =
            cublasdx::detail::convert_cute_to_copy_kind_v<coop_copy_type_b>;
        static constexpr cublasdx::detail::copy_kind coop_copy_kind =
            cublasdx::detail::choose_most_conservative_copy_instruction(coop_copy_kind_a, coop_copy_kind_b);

        static constexpr cublasdx::detail::copy_kind copy_kind =
            is_tma_possible ? cublasdx::detail::copy_kind::bulk : coop_copy_kind;

        // Descriptor type aliases using conditional tilers
        using descriptor_a_t = cute::remove_cvref_t<
            decltype(cublasdx::detail::make_gmem_descriptor<copy_kind, BLAS::max_threads_per_block, lowest_alignment>(
                COMMONDX_STL_NAMESPACE::declval<AGmemTensor>(),
                COMMONDX_STL_NAMESPACE::declval<layout_a_t>(),
                a_tiler))>;
        using descriptor_b_t = cute::remove_cvref_t<
            decltype(cublasdx::detail::make_gmem_descriptor<copy_kind, BLAS::max_threads_per_block, lowest_alignment>(
                COMMONDX_STL_NAMESPACE::declval<BGmemTensor>(),
                COMMONDX_STL_NAMESPACE::declval<layout_b_t>(),
                b_tiler))>;

        // Static functions to create descriptors
        static CUBLASDX_PIPELINE_EXECUTION descriptor_a_t make_descriptor_a(const AGmemTensor& tensor_a,
                                                                            const SmemLayoutA& smem_layout_a) {
            return cublasdx::detail::make_gmem_descriptor<copy_kind, BLAS::max_threads_per_block, lowest_alignment>(
                tensor_a, detail::get_cute_layout(smem_layout_a), a_tiler_t {});
        }

        static CUBLASDX_PIPELINE_EXECUTION descriptor_b_t make_descriptor_b(const BGmemTensor& tensor_b,
                                                                            const SmemLayoutB& smem_layout_b) {
            return cublasdx::detail::make_gmem_descriptor<copy_kind, BLAS::max_threads_per_block, lowest_alignment>(
                tensor_b, detail::get_cute_layout(smem_layout_b), b_tiler_t {});
        }
    };


    static constexpr detail::blocksize_strategy fixed_blocksize     = detail::blocksize_strategy::fixed;
    static constexpr detail::blocksize_strategy heuristic_blocksize = detail::blocksize_strategy::heuristic;

    // Prepare pipeline for staged loading from gmem to smem
    template<int PipelineDepth,
             class BLAS,
             result_storage             ResultStorage     = internal_accumulation,
             detail::blocksize_strategy BlocksizeStrategy = heuristic_blocksize,
             class AGmemTensor,
             class SmemLayoutA,
             class BGmemTensor,
             class SmemLayoutB>
    CUBLASDX_PIPELINE_EXECUTION auto make_pipeline(AGmemTensor const& a_gmem_tensor,
                                                   SmemLayoutA        smem_layout_a,
                                                   BGmemTensor const& b_gmem_tensor,
                                                   SmemLayoutB        smem_layout_b) {

        static_assert(PipelineDepth <= 16, "PipelineDepth can't exceed 16");
        static_assert(PipelineDepth > 0, "PipelineDepth must be at least 1");

        static_assert(decltype(cute::depth(a_gmem_tensor))::value == 1,
                      "a_gmem_tensor must not be a hierarchical tensor");
        static constexpr int a_rank = decltype(cute::rank(a_gmem_tensor))::value;
        static_assert(a_rank == 2 or a_rank == 3, "a_gmem_tensor must be a 2D or 3D tensor");
        using stride_0_a = decltype(cute::stride<0>(a_gmem_tensor));
        using stride_1_a = decltype(cute::stride<1>(a_gmem_tensor));
        static_assert(cute::is_constant<1, stride_0_a>::value or cute::is_constant<1, stride_1_a>::value,
                      "a_gmem_tensor must be either row or column major");

        static_assert(decltype(cute::depth(b_gmem_tensor))::value == 1,
                      "b_gmem_tensor must not be a hierarchical tensor");
        static constexpr int b_rank = decltype(cute::rank(b_gmem_tensor))::value;
        static_assert(b_rank == 2 or b_rank == 3, "b_gmem_tensor must be a 2D or 3D tensor");
        using stride_0_b = decltype(cute::stride<0>(b_gmem_tensor));
        using stride_1_b = decltype(cute::stride<1>(b_gmem_tensor));
        static_assert(cute::is_constant<1, stride_0_b>::value or cute::is_constant<1, stride_1_b>::value,
                      "b_gmem_tensor must be either row or column major");

        constexpr bool is_pipeline_descriptor =
            cublasdx::detail::has_operator<cublasdx::operator_type::with_pipeline, BLAS>::value;

        static_assert(is_pipeline_descriptor, "To be used with pipelines, you need to add WithPipeline operator");

        // Use stateless pipeline_helper struct for type computations
        constexpr int tile_m = size_of_v_m<BLAS>;
        constexpr int tile_n = size_of_v_n<BLAS>;
        constexpr int tile_k = size_of_v_k<BLAS>;

        unsigned a_m = cute::shape<0>(a_gmem_tensor);
        unsigned a_k = cute::shape<1>(a_gmem_tensor);
        unsigned b_k = cute::shape<0>(b_gmem_tensor);
        unsigned b_n = cute::shape<1>(b_gmem_tensor);

        unsigned const m = a_m;
        unsigned const n = b_n;
        unsigned const k = a_k;

        // 1. Use pipeline_helper as a type trait struct
        using pipeline_state_t =
            pipeline_helper<BLAS, AGmemTensor, BGmemTensor, SmemLayoutA, SmemLayoutB, BlocksizeStrategy>;

        static_assert(pipeline_state_t::is_smem_arrangement_same_as_gmem_a,
                      "GMEM arrangement for A must match SMEM arrangement provided by Arrangement<> operator");
        static_assert(pipeline_state_t::is_smem_arrangement_same_as_gmem_b,
                      "GMEM arrangement for B must match SMEM arrangement provided by Arrangement<> operator");

        using device_pipeline_t = detail::device_pipeline<PipelineDepth,
                                                           ResultStorage,
                                                           pipeline_state_t::disable_warp_specialization,
                                                           pipeline_state_t::copy_kind,
                                                           BLAS,
                                                           typename pipeline_state_t::descriptor_a_t,
                                                           typename pipeline_state_t::descriptor_b_t,
                                                           typename pipeline_state_t::layout_a_t,
                                                           typename pipeline_state_t::io_type_a,
                                                           typename pipeline_state_t::layout_b_t,
                                                           typename pipeline_state_t::io_type_b>;
        using host_pipeline_t   = detail::host_pipeline<device_pipeline_t>;
        using return_t          = detail::expected<host_pipeline_t, pipeline_error>;

        const bool compatible_k_dimensions = a_k == b_k;
        const bool tile_divisibility = (m % tile_m == 0 and n % tile_n == 0 and k % tile_k == 0);
        const bool enough_stages = (static_cast<int>(k / tile_k) >= static_cast<int>(PipelineDepth));

        if (!compatible_k_dimensions || !tile_divisibility || !enough_stages) {
            if (!compatible_k_dimensions) {
                return return_t {pipeline_error {pipeline_error_code::incompatible_k_dimensions}};
            }
            if (!tile_divisibility) {
                return return_t {pipeline_error {pipeline_error_code::tile_divisibility}};
            }
            return return_t {pipeline_error {pipeline_error_code::insufficient_stages}};
        }

        // 2. Create descriptors and layouts using static functions
        auto       descriptor_a = pipeline_state_t::make_descriptor_a(a_gmem_tensor, smem_layout_a);
        auto       descriptor_b = pipeline_state_t::make_descriptor_b(b_gmem_tensor, smem_layout_b);
        const auto layout_a     = detail::get_cute_layout(smem_layout_a);
        const auto layout_b     = detail::get_cute_layout(smem_layout_b);

        // 3. Create pipeline object

        auto pipeline = device_pipeline_t(descriptor_a, descriptor_b, layout_a, layout_b);

        return return_t(host_pipeline_t(pipeline));

    }

    namespace detail {
        template<int PipelineDepth,
                 class BLAS,
                 result_storage     ResultStorage,
                 blocksize_strategy BlocksizeStrategy,
                 class AGmemEngine,
                 class AGmemLayout,
                 class BGmemEngine,
                 class BGmemLayout,
                 class CudaStream = int>
        CUBLASDX_PIPELINE_EXECUTION auto suggest_emulation_pipeline(
            cublasdx::tensor<AGmemEngine, AGmemLayout> const& a_gmem_tensor,
            cublasdx::tensor<BGmemEngine, BGmemLayout> const& b_gmem_tensor,
            CudaStream const stream = 0);

        template<class BLAS,
                 result_storage     ResultStorage,
                 blocksize_strategy BlocksizeStrategy,
                 class AGmemEngine,
                 class AGmemLayout,
                 class BGmemEngine,
                 class BGmemLayout,
                 class CudaStream = int>
        CUBLASDX_PIPELINE_EXECUTION auto suggest_emulation_pipeline(
            cublasdx::tensor<AGmemEngine, AGmemLayout> const& a_gmem_tensor,
            cublasdx::tensor<BGmemEngine, BGmemLayout> const& b_gmem_tensor,
            CudaStream const stream = 0);
    } // namespace detail

    // Prepare pipeline for staged loading from gmem to smem
    template<int PipelineDepth,
             class BLAS,
             result_storage             ResultStorage     = internal_accumulation,
             detail::blocksize_strategy BlocksizeStrategy = heuristic_blocksize,
             class AGmemEngine,
             class AGmemLayout,
             class BGmemEngine,
             class BGmemLayout,
             class CudaStream = int>
    CUBLASDX_PIPELINE_EXECUTION auto suggest_pipeline(
        cublasdx::tensor<AGmemEngine, AGmemLayout> const& a_gmem_tensor,
        cublasdx::tensor<BGmemEngine, BGmemLayout> const& b_gmem_tensor,
        CudaStream const stream = 0) {
        if constexpr (has_emulation_slice_v<BLAS>) {
            return detail::suggest_emulation_pipeline<PipelineDepth, BLAS, ResultStorage, BlocksizeStrategy>(
                a_gmem_tensor, b_gmem_tensor, stream);
        } else {
            static_assert(PipelineDepth <= 16, "PipelineDepth can't exceed 16");
            static_assert(PipelineDepth > 0, "PipelineDepth must be at least 1");
            return make_pipeline<PipelineDepth, BLAS, ResultStorage, BlocksizeStrategy>(
                a_gmem_tensor,
                detail::get_cute_layout(BLAS::suggest_layout_smem_a()),
                b_gmem_tensor,
                detail::get_cute_layout(BLAS::suggest_layout_smem_b()));
        }
    }

    // Prepare pipeline for staged loading from gmem to smem
    template<class BLAS,
             result_storage             ResultStorage     = internal_accumulation,
             detail::blocksize_strategy BlocksizeStrategy = heuristic_blocksize,
             class AGmemEngine,
             class AGmemLayout,
             class BGmemEngine,
             class BGmemLayout,
             class CudaStream = int>
    CUBLASDX_PIPELINE_EXECUTION auto suggest_pipeline(
        cublasdx::tensor<AGmemEngine, AGmemLayout> const& a_gmem_tensor,
        cublasdx::tensor<BGmemEngine, BGmemLayout> const& b_gmem_tensor,
        CudaStream const stream = 0) {
        if constexpr (has_emulation_slice_v<BLAS>) {
            return detail::suggest_emulation_pipeline<BLAS, ResultStorage, BlocksizeStrategy>(
                a_gmem_tensor, b_gmem_tensor, stream);
        } else {
            auto const smem_layout_a = detail::get_cute_layout(BLAS::suggest_layout_smem_a());
            auto const smem_layout_b = detail::get_cute_layout(BLAS::suggest_layout_smem_b());

            constexpr auto pipeline_depth = suggest_max_pipeline_depth<BLAS,
                                                                       typename AGmemEngine::value_type,
                                                                       typename BGmemEngine::value_type>();

            static_assert(pipeline_depth > 0, "Not enough shared memory for pipeline with this tile");

            return make_pipeline<pipeline_depth, BLAS, ResultStorage, BlocksizeStrategy>(
                a_gmem_tensor,
                smem_layout_a,
                b_gmem_tensor,
                smem_layout_b);
        }
    }
} // namespace cublasdx

#endif // defined(CUBLASDX_OVERLOAD_DEVICE_PIPELINE_CREATION) || !defined(COMMONDX_DETAIL_USE_CUDA_STL)

#endif // CUBLASDX_DETAIL_PIPELINE_HPP
