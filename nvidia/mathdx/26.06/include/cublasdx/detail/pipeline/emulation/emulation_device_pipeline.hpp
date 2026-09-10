// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_DEVICE_PIPELINE_HPP
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_DEVICE_PIPELINE_HPP

#include <cuda_runtime_api.h>

#include "cublasdx/detail/copy.hpp"
#include "cublasdx/detail/commondx_config.hpp"
#include "cublasdx/detail/pipeline/emulation/emulation_tile_pipeline.hpp"
#include "cublasdx/detail/pipeline/pipeline_result.hpp"
#include "cublasdx/detail/shared_memory.hpp"
#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/traits.hpp"

#ifdef COMMONDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/cstdint>
#else
#    include <cstdint>
#endif

namespace cublasdx {
    namespace detail {
        inline constexpr unsigned emulation_shift_alignment = 16;
        inline constexpr unsigned emulation_temporary_alignment = 256;

        template<class ExternalBLAS,
                 class InternalBLAS,
                 class InternalDevicePipeline,
                 class AShiftTensor,
                 class BShiftTensor,
                 int Slices>
        struct emulation_device_pipeline {
            using external_blas_t = ExternalBLAS;
            using internal_blas_t = InternalBLAS;
            using internal_device_pipeline_t = InternalDevicePipeline;

            static constexpr auto mma_instruction_kind = InternalDevicePipeline::mma_instruction_kind;
            static constexpr bool is_warp_specialized = InternalDevicePipeline::is_warp_specialized;
            static constexpr int additional_producer_threads = InternalDevicePipeline::additional_producer_threads;
            static constexpr int max_threads_per_block = InternalDevicePipeline::max_threads_per_block;
            static constexpr int first_producer_warp_idx = InternalDevicePipeline::first_producer_warp_idx;
            static constexpr unsigned alignment_a = InternalDevicePipeline::alignment_a;
            static constexpr unsigned alignment_b = InternalDevicePipeline::alignment_b;
            static constexpr int tile_m = size_of_v_m<ExternalBLAS>;
            static constexpr int tile_n = size_of_v_n<ExternalBLAS>;
            static constexpr int tile_k = size_of_v_k<ExternalBLAS>;
            static constexpr unsigned shift_alignment = emulation_shift_alignment;

            InternalDevicePipeline internal_pipeline;
            AShiftTensor shift_a;
            BShiftTensor shift_b;
            int slice_count = Slices;
            CUBLASDX_DEVICE_PIPELINE_EXECUTION constexpr int stages() const {
                return internal_pipeline.stages();
            }

            CUBLASDX_DEVICE_PIPELINE_EXECUTION constexpr dim3 get_block_dim() const {
                return internal_pipeline.get_block_dim();
            }

            CUBLASDX_DEVICE_PIPELINE_EXECUTION constexpr auto buffer_alignment() const {
                return cute::max(internal_pipeline.buffer_alignment(), shift_alignment);
            }

            CUBLASDX_DEVICE_PIPELINE_EXECUTION unsigned buffer_size() const {
                return cublasdx::make_shared_storage_calculator()
                    .add(internal_pipeline.buffer_alignment(), internal_pipeline.buffer_size())
                    .add(shift_alignment, sizeof(COMMONDX_STL_NAMESPACE::int32_t), tile_m)
                    .add(shift_alignment, sizeof(COMMONDX_STL_NAMESPACE::int32_t), tile_n)
                    .get();
            }

        private:
            CUBLASDX_DEVICE auto make_smem_shift_layout_a() const {
                return cute::make_layout(cute::make_shape(cute::Int<tile_m> {}), cute::make_stride(cute::_1 {}));
            }

            CUBLASDX_DEVICE auto make_smem_shift_layout_b() const {
                return cute::make_layout(cute::make_shape(cute::Int<tile_n> {}), cute::make_stride(cute::_1 {}));
            }

            template<class CoordA, class CoordB, class SmemShiftA, class SmemShiftB>
            CUBLASDX_DEVICE void load_shift_tiles(CoordA const& coord_a,
                                                  CoordB const& coord_b,
                                                  SmemShiftA& smem_shift_a,
                                                  SmemShiftB& smem_shift_b) const {
                auto const output_tile_m = cute::get<0>(coord_a);
                auto const output_tile_n = cute::get<0>(coord_b);
                cublasdx::copy<InternalBLAS, shift_alignment>(shift_a(cublasdx::slice, output_tile_m), smem_shift_a);
                cublasdx::copy<InternalBLAS, shift_alignment>(shift_b(cublasdx::slice, output_tile_n), smem_shift_b);
                cublasdx::copy_wait();
            }

        public:
            template<class CoordA, class CoordB>
            CUBLASDX_DEVICE auto get_tile(void* smem, CoordA const& coord_a, CoordB const& coord_b) const {
                auto smem_shift_layout_a = make_smem_shift_layout_a();
                auto smem_shift_layout_b = make_smem_shift_layout_b();

                auto slicing_results = cublasdx::shared_memory::slice<char,
                                                                       COMMONDX_STL_NAMESPACE::int32_t,
                                                                       COMMONDX_STL_NAMESPACE::int32_t>(
                    smem,
                    internal_pipeline.buffer_alignment(),
                    internal_pipeline.buffer_size(),
                    shift_alignment,
                    smem_shift_layout_a,
                    shift_alignment,
                    smem_shift_layout_b);

                auto pipeline_smem = cute::get<0>(slicing_results);
                auto smem_shift_a = cute::get<1>(slicing_results);
                auto smem_shift_b = cute::get<2>(slicing_results);

                load_shift_tiles(coord_a, coord_b, smem_shift_a, smem_shift_b);

                auto const output_tile_m = cute::get<0>(coord_a);
                auto const output_tile_n = cute::get<0>(coord_b);
                auto internal_coord_a = cublasdx::make_coord(output_tile_m, 0);
                auto internal_coord_b = cublasdx::make_coord(output_tile_n, slice_count - 1);

                using internal_tile_t = cute::remove_cvref_t<decltype(
                    internal_pipeline.get_tile(pipeline_smem, internal_coord_a, internal_coord_b))>;
                using smem_shift_a_t = cute::remove_cvref_t<decltype(smem_shift_a)>;
                using smem_shift_b_t = cute::remove_cvref_t<decltype(smem_shift_b)>;
                return emulation_tile_pipeline<ExternalBLAS,
                                               InternalBLAS,
                                               InternalDevicePipeline,
                                               internal_tile_t,
                                               smem_shift_a_t,
                                               smem_shift_b_t,
                                               Slices>(internal_pipeline,
                                                       pipeline_smem,
                                                       internal_coord_a,
                                                       internal_coord_b,
                                                       smem_shift_a,
                                                       smem_shift_b,
                                                       output_tile_m,
                                                       output_tile_n,
                                                       slice_count);
            }

            template<class TilePipeline, class CoordA, class CoordB>
            CUBLASDX_DEVICE void reset_tile(TilePipeline& tile_pipeline, CoordA const& coord_a, CoordB const& coord_b) const {
                tile_pipeline.output_tile_m = cute::get<0>(coord_a);
                tile_pipeline.output_tile_n = cute::get<0>(coord_b);
                tile_pipeline.slice_count = slice_count;

                // The single smem shift buffer is read only by the internal-tile
                // epilogue reconstruction. Load it here from the epilogue threads,
                // bracketed by epilogue_sync(): the leading sync waits until the
                // previous tile's reconstruction has finished reading the buffer, and
                // the trailing fence/wait/sync publishes the new load before the next
                // reconstruction reads it. Same warps do the load and the read, so
                // one buffer is race-free under persistent tile reuse.
                if (tile_pipeline.internal_tile_pipeline.is_active()) {
                    constexpr unsigned num_epilogue_threads =
                        TilePipeline::internal_tile_pipeline_t::num_epilogue_threads;
                    const unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
                    tile_pipeline.epilogue_sync();
                    cublasdx::copy<num_epilogue_threads, shift_alignment>(
                        tid, shift_a(cublasdx::slice, tile_pipeline.output_tile_m), tile_pipeline.smem_shift_a);
                    cublasdx::copy<num_epilogue_threads, shift_alignment>(
                        tid, shift_b(cublasdx::slice, tile_pipeline.output_tile_n), tile_pipeline.smem_shift_b);
                    cute::cp_async_fence();
                    cute::cp_async_wait<0>();
                    tile_pipeline.epilogue_sync();
                }

                internal_pipeline.reset_tile(tile_pipeline.internal_tile_pipeline,
                                             cublasdx::make_coord(tile_pipeline.output_tile_m, 0),
                                             cublasdx::make_coord(tile_pipeline.output_tile_n, slice_count - 1));
            }
        };

        template<class DevicePipeline>
        class host_emulation_pipeline;

        template<class ExternalBLAS,
                 class InternalBLAS,
                 class InternalDevicePipeline,
                 class AShiftTensor,
                 class BShiftTensor,
                 int Slices>
        class host_emulation_pipeline<emulation_device_pipeline<ExternalBLAS,
                                                                InternalBLAS,
                                                                InternalDevicePipeline,
                                                                AShiftTensor,
                                                                BShiftTensor,
                                                                Slices>> {
            using emulation_pipeline_type = emulation_device_pipeline<ExternalBLAS,
                                                                      InternalBLAS,
                                                                      InternalDevicePipeline,
                                                                      AShiftTensor,
                                                                      BShiftTensor,
                                                                      Slices>;

            emulation_pipeline_type device_pipeline_;
            void* allocation_base_ = nullptr;
            cudaStream_t stream_ = 0;

        public:
            using device_pipeline_type = emulation_pipeline_type;
            static constexpr int max_threads_per_block = device_pipeline_type::max_threads_per_block;

            host_emulation_pipeline(device_pipeline_type const& device_pipeline,
                          void* allocation_base,
                          cudaStream_t const stream = 0):
                device_pipeline_(device_pipeline),
                allocation_base_(allocation_base),
                stream_(stream) {}

            host_emulation_pipeline(host_emulation_pipeline const&) = delete;
            host_emulation_pipeline& operator=(host_emulation_pipeline const&) = delete;

            host_emulation_pipeline(host_emulation_pipeline&& other) noexcept:
                device_pipeline_(other.device_pipeline_),
                allocation_base_(other.allocation_base_),
                stream_(other.stream_) {
                other.allocation_base_ = nullptr;
            }

            host_emulation_pipeline& operator=(host_emulation_pipeline&& other) noexcept {
                if (this != &other) {
                    release();
                    device_pipeline_ = other.device_pipeline_;
                    allocation_base_ = other.allocation_base_;
                    stream_ = other.stream_;
                    other.allocation_base_ = nullptr;
                }
                return *this;
            }

            ~host_emulation_pipeline() {
                release();
            }

            cudaError_t release() {
                cudaError_t result = cudaSuccess;
                if (allocation_base_ != nullptr) {
                    result = cudaFreeAsync(allocation_base_, stream_);
                    allocation_base_ = nullptr;
                }
                return result;
            }

            device_pipeline_type get_device_handle() const& {
                return device_pipeline_;
            }

            device_pipeline_type get_device_handle() && = delete;

            constexpr int stages() const {
                return device_pipeline_.stages();
            }

            constexpr auto get_block_dim() const {
                return device_pipeline_.get_block_dim();
            }

            constexpr auto buffer_alignment() const {
                return device_pipeline_.buffer_alignment();
            }

            auto buffer_size() const {
                return device_pipeline_.buffer_size();
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_EMULATION_DEVICE_PIPELINE_HPP
