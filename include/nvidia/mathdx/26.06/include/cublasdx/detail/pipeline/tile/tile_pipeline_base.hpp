// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_BASE_HPP
#define CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_BASE_HPP

#include <cutlass/pipeline/sm90_pipeline.hpp>

#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/detail/pipeline/logging.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_stage.hpp"

namespace cublasdx {
    namespace detail {

        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 class BLAS,
                 class GmemDescriptorA,
                 class CopyFunctorA,
                 class GmemDescriptorB,
                 class CopyFunctorB,
                 class PipelinedSmemLayoutA,
                 typename IOTypeA,
                 class PipelinedSmemLayoutB,
                 typename IOTypeB>
        struct tile_pipeline_base {
            static_assert(cute::is_layout<PipelinedSmemLayoutA>::value, "PipelinedSmemLayoutA must be a cute::Layout");
            static_assert(cute::is_layout<PipelinedSmemLayoutB>::value, "PipelinedSmemLayoutB must be a cute::Layout");

            static constexpr result_storage accumulation = validate_accumulator_mode<ResultStorage>::value;

            static_assert((cute::size(PipelinedSmemLayoutA {}) % PipelineDepth) == 0);
            static_assert((cute::size(PipelinedSmemLayoutB {}) % PipelineDepth) == 0);
            static constexpr unsigned copy_bytes =
                (cute::size(PipelinedSmemLayoutA {}) * sizeof(typename BLAS::a_value_type) +
                 cute::size(PipelinedSmemLayoutB {}) * sizeof(typename BLAS::b_value_type)) /
                PipelineDepth;

            static constexpr unsigned sm = sm_of_v<BLAS>;

            static constexpr bool has_static_block_dim = cublasdx::has_static_block_dim_v<BLAS>;

            static constexpr int tile_n = size_of_v_n<BLAS>;

            static constexpr int rank = decltype(cute::rank(GmemDescriptorA {}))::value - 1;

            // --- Shared data members ---

            unsigned const tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
            unsigned const warp_idx = tid / 32;

            pipeline_stage_cursor<PipelineDepth> read_cursor {};
            pipeline_stage_cursor<PipelineDepth> write_cursor =
                pipeline_stage_cursor<PipelineDepth>::producer_start();

            unsigned k_chunks;

            GmemDescriptorA gmem_a;
            CopyFunctorA    copy_functor_a;
            using a_value_type = typename BLAS::a_value_type;
            using a_smem_tensor_t =
                decltype(cublasdx::make_tensor(cute::make_smem_ptr<IOTypeA>(nullptr), PipelinedSmemLayoutA {}));

            GmemDescriptorB gmem_b;
            CopyFunctorB    copy_functor_b;
            using b_value_type = typename BLAS::b_value_type;
            using b_smem_tensor_t =
                decltype(cublasdx::make_tensor(cute::make_smem_ptr<IOTypeB>(nullptr), PipelinedSmemLayoutB {}));
            pipeline_shared_tensor_storage<a_smem_tensor_t, b_smem_tensor_t> shared_tensor_storage;

            // --- Constructor ---
            CUBLASDX_DEVICE
            tile_pipeline_base(int count, GmemDescriptorA const& in_gmem_a, CopyFunctorA copy_a,
                          GmemDescriptorB const& in_gmem_b, CopyFunctorB copy_b):
                k_chunks(count), gmem_a(in_gmem_a), copy_functor_a(copy_a),
                gmem_b(in_gmem_b), copy_functor_b(copy_b) {}

            // --- Shared getters ---

            CUBLASDX_DEVICE
            constexpr auto stages() { return cute::Int<PipelineDepth> {}; }

            CUBLASDX_DEVICE
            unsigned compute_smem_stage() const { return read_cursor.index(); }

            CUBLASDX_DEVICE
            unsigned copy_smem_stage() const { return write_cursor.index(); }

            CUBLASDX_DEVICE
            auto compute_smem_tensors() const {
                return shared_tensor_storage.compute_smem_tensors(read_cursor.index());
            }

            CUBLASDX_DEVICE
            auto copy_smem_tensors() const {
                return shared_tensor_storage.copy_smem_tensors(write_cursor.index());
            }

            CUBLASDX_DEVICE
            unsigned chunks() const { return k_chunks; }

            CUBLASDX_DEVICE
            unsigned compute_smem_chunk() const { return read_cursor.chunk; }

            CUBLASDX_DEVICE
            unsigned copy_smem_chunk() const { return write_cursor.chunk; }

            CUBLASDX_DEVICE
            bool has_copy_smem_work() const { return write_cursor.has_work(k_chunks); }

            CUBLASDX_DEVICE
            bool has_compute_smem_work() const { return read_cursor.has_work(k_chunks); }

            // --- Tile reset ---

            CUBLASDX_DEVICE
            void reset_tile() {
                PIPELINE_LOG("tile reset: %d\n", threadIdx.x);
                write_cursor.reset_chunk();
                read_cursor.reset_chunk();
            }

            CUBLASDX_DEVICE
            void reset_tile(GmemDescriptorA const& reset_row_a, GmemDescriptorB const& reset_col_b) {
                PIPELINE_LOG("tile reset: %d\n", threadIdx.x);
                gmem_a = reset_row_a;
                gmem_b = reset_col_b;
                write_cursor.reset_chunk();
                read_cursor.reset_chunk();
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_BASE_HPP
