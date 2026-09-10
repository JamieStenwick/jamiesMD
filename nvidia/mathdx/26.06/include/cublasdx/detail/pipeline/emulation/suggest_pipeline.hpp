// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_SUGGEST_PIPELINE_HPP
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_SUGGEST_PIPELINE_HPP

#include <cassert>
#include <cstddef>
#include <cuda_runtime_api.h>

#include "commondx/traits/dx_traits.hpp"

#include "cublasdx/detail/commondx_config.hpp"
#include "cublasdx/detail/pipeline/emulation/emulation_device_pipeline.hpp"
#include "cublasdx/detail/pipeline/emulation/emulation_mempool.hpp"
#include "cublasdx/detail/pipeline/emulation/slicing.hpp"
#include "cublasdx/detail/pipeline/pipeline.hpp"
#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/traits.hpp"

#ifdef COMMONDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/cstdint>
#    include <cuda/std/type_traits>
#else
#    include <cstdint>
#    include <type_traits>
#endif

namespace cublasdx {
    namespace detail {
        template<class T>
        static constexpr bool is_supported_emulation_input_precision_v =
            COMMONDX_STL_NAMESPACE::is_same_v<T, float> || COMMONDX_STL_NAMESPACE::is_same_v<T, double>;

        template<class BLAS>
        struct emulation_internal_blas {
        private:
            using without_external_precision = commondx::detail::filter_t<BLAS, operator_type, operator_type::precision>;
            using without_required_mantissa_bits =
                commondx::detail::filter_t<without_external_precision,
                                            operator_type,
                                            operator_type::required_mantissa_bits>;

        public:
            using type = commondx::detail::concatenate_description_t<Precision<COMMONDX_STL_NAMESPACE::int8_t,
                                                                                COMMONDX_STL_NAMESPACE::int8_t,
                                                                                COMMONDX_STL_NAMESPACE::int32_t>,
                                                                     without_required_mantissa_bits>;
        };

        template<class BLAS>
        using emulation_internal_blas_t = typename emulation_internal_blas<BLAS>::type;

        template<class CudaStream>
        CUBLASDX_PIPELINE_EXECUTION cudaStream_t emulation_get_cuda_stream(CudaStream const stream) {
            static_assert(COMMONDX_STL_NAMESPACE::is_same_v<CudaStream, cudaStream_t> ||
                              COMMONDX_STL_NAMESPACE::is_same_v<CudaStream, int>,
                          "The emulation pipeline stream argument must be cudaStream_t or the default int value 0");
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CudaStream, int>) {
                assert(stream == 0 && "The emulation pipeline int stream argument must be 0");
                return cudaStream_t {};
            } else {
                return stream;
            }
        }

        template<class BLAS>
        CUBLASDX_PIPELINE_EXECUTION constexpr unsigned emulation_pipeline_shift_smem_size() {
            constexpr unsigned shift_alignment = emulation_shift_alignment;
            unsigned size = 0;
            size = cutlass::round_up(size, shift_alignment) +
                   cublasdx::size_of_v_m<BLAS> * sizeof(COMMONDX_STL_NAMESPACE::int32_t);
            size = cutlass::round_up(size, shift_alignment) +
                   cublasdx::size_of_v_n<BLAS> * sizeof(COMMONDX_STL_NAMESPACE::int32_t);
            return size;
        }

        inline constexpr COMMONDX_STL_NAMESPACE::uint64_t emulation_max_slice_value =
            (static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(1) << 8) - 1;
        inline constexpr unsigned emulation_max_k_without_int32_accumulator_overflow =
            static_cast<unsigned>(((static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(1) << 31) - 1) /
                                  (emulation_max_slice_value * emulation_max_slice_value));

        CUBLASDX_PIPELINE_EXECUTION constexpr COMMONDX_STL_NAMESPACE::uint64_t
        emulation_round_up(COMMONDX_STL_NAMESPACE::uint64_t const value, unsigned const alignment) {
            auto const alignment_u64 = static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(alignment);
            return ((value + alignment_u64 - 1) / alignment_u64) * alignment_u64;
        }

        CUBLASDX_PIPELINE_EXECUTION constexpr COMMONDX_STL_NAMESPACE::uint64_t
        add_emulation_temporary_extent(COMMONDX_STL_NAMESPACE::uint64_t const current,
                                       unsigned const alignment,
                                       COMMONDX_STL_NAMESPACE::uint64_t const elem_size,
                                       COMMONDX_STL_NAMESPACE::uint64_t const num_elements) {
            return emulation_round_up(current, alignment) + elem_size * num_elements;
        }

        template<class T>
        CUBLASDX_PIPELINE_EXECUTION T*
        slice_emulation_temporary_pointer(void* base,
                                          COMMONDX_STL_NAMESPACE::uint64_t& offset,
                                          unsigned const alignment,
                                          COMMONDX_STL_NAMESPACE::uint64_t const num_elements) {
            assert(alignment % alignof(T) == 0);
            assert(alignment >= alignof(T));
            offset = emulation_round_up(offset, alignment);
            auto* ptr = reinterpret_cast<T*>(static_cast<char*>(base) + offset);
            offset += static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(sizeof(T)) * num_elements;
            return ptr;
        }

        template<class BLAS, class InternalBLAS, class ValueTypeA, class ValueTypeB>
        CUBLASDX_PIPELINE_EXECUTION constexpr int suggest_emulation_max_pipeline_depth() {
            constexpr unsigned max_alignment = 128;
            constexpr unsigned stage_shared_req =
                cutlass::round_up(
                    cublasdx::cosize(decltype(detail::get_cute_layout(InternalBLAS::suggest_layout_smem_a())) {}) *
                        sizeof(ValueTypeA),
                    max_alignment) +
                cutlass::round_up(
                    cublasdx::cosize(decltype(detail::get_cute_layout(InternalBLAS::suggest_layout_smem_b())) {}) *
                        sizeof(ValueTypeB),
                    max_alignment) +
                sizeof(cublasdx::pipeline_stage_scratch_t);

            constexpr unsigned shared_memory = commondx::device_info<sm_of_v<InternalBLAS>>::shared_memory();
            constexpr unsigned reserved_shared_memory =
                32 + (emulation_shift_alignment - 1) + emulation_pipeline_shift_smem_size<BLAS>();
            static_assert(shared_memory > reserved_shared_memory,
                          "Not enough shared memory for emulation pipeline metadata");
            return cute::min(16, (shared_memory - reserved_shared_memory) / stage_shared_req);
        }


        template<int PipelineDepth,
                 class BLAS,
                 result_storage     ResultStorage,
                 blocksize_strategy BlocksizeStrategy,
                 class AGmemEngine,
                 class AGmemLayout,
                 class BGmemEngine,
                 class BGmemLayout,
                 class CudaStream>
        CUBLASDX_PIPELINE_EXECUTION auto suggest_emulation_pipeline(
            cublasdx::tensor<AGmemEngine, AGmemLayout> const& a_gmem_tensor,
            cublasdx::tensor<BGmemEngine, BGmemLayout> const& b_gmem_tensor,
            CudaStream const stream) {
            cudaStream_t const cuda_stream = emulation_get_cuda_stream(stream);
            static_assert(PipelineDepth <= 16, "PipelineDepth can't exceed 16");
            static_assert(PipelineDepth > 0, "PipelineDepth must be at least 1");
            static_assert(ResultStorage == reusable_accumulator,
                          "RequiredMantissaBits<X>() pipelines currently require cublasdx::reusable_accumulator");

            static constexpr int rank_a = decltype(cute::rank(a_gmem_tensor))::value;
            static constexpr int rank_b = decltype(cute::rank(b_gmem_tensor))::value;
            static_assert(rank_a == 2 && rank_b == 2,
                          "RequiredMantissaBits<X>() requires rank-2 A and B tensors. This restriction will be lifted in future releases");

            using a_input_t = cute::remove_cvref_t<typename AGmemEngine::value_type>;
            using b_input_t = cute::remove_cvref_t<typename BGmemEngine::value_type>;
            static_assert(COMMONDX_STL_NAMESPACE::is_same_v<a_input_t, precision_of_a_t<BLAS>> &&
                              COMMONDX_STL_NAMESPACE::is_same_v<b_input_t, precision_of_b_t<BLAS>> &&
                              is_supported_emulation_input_precision_v<a_input_t> &&
                              is_supported_emulation_input_precision_v<b_input_t>,
                          "RequiredMantissaBits<X>() requires float or double input tensors matching BLAS precision");

            constexpr int slices = cublasdx::emulation_slice_of_v<BLAS>;
            using internal_blas_t = emulation_internal_blas_t<BLAS>;
            using slice_tensor_a_t = decltype(make_emulation_slice_tensor<cublasdx::arrangement_of_v_a<BLAS>, slices>(
                static_cast<COMMONDX_STL_NAMESPACE::int8_t*>(nullptr), 0u, 0u));
            using slice_tensor_b_t = decltype(make_emulation_slice_tensor<cublasdx::arrangement_of_v_b<BLAS>, slices>(
                static_cast<COMMONDX_STL_NAMESPACE::int8_t*>(nullptr), 0u, 0u));
            using shift_tensor_a_t = decltype(make_emulation_shift_tensor<cublasdx::size_of_v_m<BLAS>>(
                static_cast<COMMONDX_STL_NAMESPACE::int32_t*>(nullptr), 0u));
            using shift_tensor_b_t = decltype(make_emulation_shift_tensor<cublasdx::size_of_v_n<BLAS>>(
                static_cast<COMMONDX_STL_NAMESPACE::int32_t*>(nullptr), 0u));
            using smem_layout_a_t = decltype(get_cute_layout(internal_blas_t::suggest_layout_smem_a()));
            using smem_layout_b_t = decltype(get_cute_layout(internal_blas_t::suggest_layout_smem_b()));
            using internal_pipeline_result_t = decltype(cublasdx::make_pipeline<PipelineDepth,
                                                                             internal_blas_t,
                                                                             reusable_accumulator,
                                                                             BlocksizeStrategy>(
                COMMONDX_STL_NAMESPACE::declval<slice_tensor_a_t const&>(),
                COMMONDX_STL_NAMESPACE::declval<smem_layout_a_t>(),
                COMMONDX_STL_NAMESPACE::declval<slice_tensor_b_t const&>(),
                COMMONDX_STL_NAMESPACE::declval<smem_layout_b_t>()));
            using internal_host_pipeline_t = typename internal_pipeline_result_t::value_type;
            using internal_pipeline_t = typename internal_host_pipeline_t::device_pipeline_type;
            using view_t = emulation_device_pipeline<BLAS,
                                                             internal_blas_t,
                                                             internal_pipeline_t,
                                                             shift_tensor_a_t,
                                                             shift_tensor_b_t,
                                                             slices>;
            using host_emulation_pipeline_t = host_emulation_pipeline<view_t>;
            using return_t = expected<host_emulation_pipeline_t, pipeline_error>;
            using error_t = typename return_t::error_type;
            using error_code = pipeline_error_code;

            unsigned const m = cute::shape<0>(a_gmem_tensor);
            unsigned const k = cute::shape<1>(a_gmem_tensor);
            unsigned const b_k = cute::shape<0>(b_gmem_tensor);
            unsigned const n = cute::shape<1>(b_gmem_tensor);

            auto precheck_slice_tensor_a = make_emulation_slice_tensor<cublasdx::arrangement_of_v_a<BLAS>, slices>(
                static_cast<COMMONDX_STL_NAMESPACE::int8_t*>(nullptr), m, k);
            auto precheck_slice_tensor_b = make_emulation_slice_tensor<cublasdx::arrangement_of_v_b<BLAS>, slices>(
                static_cast<COMMONDX_STL_NAMESPACE::int8_t*>(nullptr), b_k, n);

            auto precheck_pipeline_result = cublasdx::make_pipeline<PipelineDepth,
                                                                 internal_blas_t,
                                                                 reusable_accumulator,
                                                                 BlocksizeStrategy>(
                precheck_slice_tensor_a,
                get_cute_layout(internal_blas_t::suggest_layout_smem_a()),
                precheck_slice_tensor_b,
                get_cute_layout(internal_blas_t::suggest_layout_smem_b()));

            if (!precheck_pipeline_result) {
                return return_t {precheck_pipeline_result.error()};
            }
            if (k > emulation_max_k_without_int32_accumulator_overflow) {
                return return_t {error_t {error_code::emulation_k_dimension_too_large}};
            }

            void* allocation_base = nullptr;
            auto free_allocation = [&]() {
                if (allocation_base != nullptr) {
                    cudaFreeAsync(allocation_base, cuda_stream);
                    allocation_base = nullptr;
                }
            };

            COMMONDX_STL_NAMESPACE::uint64_t const slice_a_elements =
                static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(m) * k * static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(slices);
            COMMONDX_STL_NAMESPACE::uint64_t const slice_b_elements =
                static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(b_k) * n * static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(slices);
            COMMONDX_STL_NAMESPACE::uint64_t const shift_a_elements = m;
            COMMONDX_STL_NAMESPACE::uint64_t const shift_b_elements = n;

            COMMONDX_STL_NAMESPACE::uint64_t allocation_size = 0;
            allocation_size = add_emulation_temporary_extent(allocation_size,
                                                             emulation_temporary_alignment,
                                                             sizeof(COMMONDX_STL_NAMESPACE::int8_t),
                                                             slice_a_elements);
            allocation_size = add_emulation_temporary_extent(allocation_size,
                                                             emulation_temporary_alignment,
                                                             sizeof(COMMONDX_STL_NAMESPACE::int8_t),
                                                             slice_b_elements);
            allocation_size = add_emulation_temporary_extent(allocation_size,
                                                             emulation_temporary_alignment,
                                                             sizeof(COMMONDX_STL_NAMESPACE::int32_t),
                                                             shift_a_elements);
            allocation_size = add_emulation_temporary_extent(allocation_size,
                                                             emulation_temporary_alignment,
                                                             sizeof(COMMONDX_STL_NAMESPACE::int32_t),
                                                             shift_b_elements);

            cudaError_t err = emulation_malloc_async(&allocation_base, static_cast<std::size_t>(allocation_size), cuda_stream);
            if (err != cudaSuccess) {
                return return_t {error_t {error_code::allocation_failed, static_cast<int>(err)}};
            }

            COMMONDX_STL_NAMESPACE::uint64_t allocation_offset = 0;
            auto* slice_a = slice_emulation_temporary_pointer<COMMONDX_STL_NAMESPACE::int8_t>(
                allocation_base, allocation_offset, emulation_temporary_alignment, slice_a_elements);
            auto* slice_b = slice_emulation_temporary_pointer<COMMONDX_STL_NAMESPACE::int8_t>(
                allocation_base, allocation_offset, emulation_temporary_alignment, slice_b_elements);
            auto* shift_a = slice_emulation_temporary_pointer<COMMONDX_STL_NAMESPACE::int32_t>(
                allocation_base, allocation_offset, emulation_temporary_alignment, shift_a_elements);
            auto* shift_b = slice_emulation_temporary_pointer<COMMONDX_STL_NAMESPACE::int32_t>(
                allocation_base, allocation_offset, emulation_temporary_alignment, shift_b_elements);

            auto slice_tensor_a = make_emulation_slice_tensor<cublasdx::arrangement_of_v_a<BLAS>, slices>(slice_a, m, k);
            auto slice_tensor_b = make_emulation_slice_tensor<cublasdx::arrangement_of_v_b<BLAS>, slices>(slice_b, b_k, n);
            auto shift_tensor_a = make_emulation_shift_tensor<cublasdx::size_of_v_m<BLAS>>(shift_a, m / cublasdx::size_of_v_m<BLAS>);
            auto shift_tensor_b = make_emulation_shift_tensor<cublasdx::size_of_v_n<BLAS>>(shift_b, n / cublasdx::size_of_v_n<BLAS>);

            auto internal_pipeline_result = cublasdx::make_pipeline<PipelineDepth,
                                                                 internal_blas_t,
                                                                 reusable_accumulator,
                                                                 BlocksizeStrategy>(
                slice_tensor_a,
                get_cute_layout(internal_blas_t::suggest_layout_smem_a()),
                slice_tensor_b,
                get_cute_layout(internal_blas_t::suggest_layout_smem_b()));

            if (!internal_pipeline_result) {
                free_allocation();
                return return_t {internal_pipeline_result.error()};
            }
            auto internal_pipeline = internal_pipeline_result->get_device_handle();

            constexpr int reduction_block_size = 128;
            emulation_max_reduce_kernel<reduction_block_size, emulation_slice_matrix::a>
                <<<m, reduction_block_size, 0, cuda_stream>>>(a_gmem_tensor, shift_tensor_a);
            emulation_max_reduce_kernel<reduction_block_size, emulation_slice_matrix::b>
                <<<n, reduction_block_size, 0, cuda_stream>>>(b_gmem_tensor, shift_tensor_b);

            constexpr int slice_block_size = 128;
            COMMONDX_STL_NAMESPACE::uint64_t const total_a = static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(m) * k;
            COMMONDX_STL_NAMESPACE::uint64_t const total_b = static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(b_k) * n;
            unsigned const blocks_a = static_cast<unsigned>((total_a + slice_block_size - 1) / slice_block_size);
            unsigned const blocks_b = static_cast<unsigned>((total_b + slice_block_size - 1) / slice_block_size);
            emulation_slice_kernel<slice_block_size, slices, emulation_slice_matrix::a>
                <<<blocks_a, slice_block_size, 0, cuda_stream>>>(a_gmem_tensor, shift_tensor_a, slice_tensor_a, k, total_a);
            emulation_slice_kernel<slice_block_size, slices, emulation_slice_matrix::b>
                <<<blocks_b, slice_block_size, 0, cuda_stream>>>(b_gmem_tensor, shift_tensor_b, slice_tensor_b, b_k, total_b);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                free_allocation();
                return return_t {error_t {static_cast<int>(err)}};
            }

            view_t view {internal_pipeline, shift_tensor_a, shift_tensor_b, slices};
            return return_t(host_emulation_pipeline_t(view, allocation_base, cuda_stream));
        }


        template<class BLAS,
                 result_storage     ResultStorage,
                 blocksize_strategy BlocksizeStrategy,
                 class AGmemEngine,
                 class AGmemLayout,
                 class BGmemEngine,
                 class BGmemLayout,
                 class CudaStream>
        CUBLASDX_PIPELINE_EXECUTION auto suggest_emulation_pipeline(
            cublasdx::tensor<AGmemEngine, AGmemLayout> const& a_gmem_tensor,
            cublasdx::tensor<BGmemEngine, BGmemLayout> const& b_gmem_tensor,
            CudaStream const stream) {
            using internal_blas_t = emulation_internal_blas_t<BLAS>;
            constexpr auto pipeline_depth = suggest_emulation_max_pipeline_depth<BLAS,
                                                                                 internal_blas_t,
                                                                                 COMMONDX_STL_NAMESPACE::int8_t,
                                                                                 COMMONDX_STL_NAMESPACE::int8_t>();

            static_assert(pipeline_depth > 0, "Not enough shared memory for emulation pipeline with this tile");

            return suggest_emulation_pipeline<pipeline_depth, BLAS, ResultStorage, BlocksizeStrategy>(
                a_gmem_tensor, b_gmem_tensor, stream);
        }


    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_EMULATION_SUGGEST_PIPELINE_HPP
