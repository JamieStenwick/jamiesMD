// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_STAGE_HPP
#define CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_STAGE_HPP

#include <cutlass/pipeline/sm90_pipeline.hpp>

#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/detail/pipeline/accumulator_mode.hpp"

namespace cublasdx {
    namespace detail {

        template<unsigned Depth>
        struct pipeline_stage_cursor {
            using state_t = cutlass::PipelineState<Depth>;

            state_t  state {};
            unsigned chunk = 0;

            CUBLASDX_DEVICE
            static pipeline_stage_cursor producer_start() {
                pipeline_stage_cursor ret {};
                ret.state = cutlass::make_producer_start_state<state_t>();
                return ret;
            }


            CUBLASDX_DEVICE
            auto index() const { return state.index(); }

            CUBLASDX_DEVICE
            auto phase() const { return state.phase(); }

            CUBLASDX_DEVICE
            bool has_work(unsigned total_chunks) const { return chunk < total_chunks; }

            CUBLASDX_DEVICE
            void advance() {
                ++state;
                ++chunk;
            }

            CUBLASDX_DEVICE
            void reset_chunk() { chunk = 0; }

            CUBLASDX_DEVICE
            auto previous_index() const {
                return (state.index() == 0) ? (Depth - 1) : (state.index() - 1);
            }

            CUBLASDX_DEVICE
            auto previous_phase() const {
                return (state.index() == 0) ? (state.phase() ^ 1) : state.phase();
            }
        };

        template<class SmemA, class SmemB>
        struct pipeline_shared_tensor_storage {
            SmemA smem_a;
            SmemB smem_b;

            CUBLASDX_DEVICE
            void reset(SmemA const& a, SmemB const& b) {
                smem_a = a;
                smem_b = b;
            }

            CUBLASDX_DEVICE
            auto compute_smem_tensors(unsigned stage) const {
                return cute::make_tuple(smem_a(cublasdx::slice, cublasdx::slice, stage),
                                        smem_b(cublasdx::slice, cublasdx::slice, stage));
            }

            CUBLASDX_DEVICE
            auto copy_smem_tensors(unsigned stage) const {
                return cute::make_tuple(smem_a(cublasdx::slice, cublasdx::slice, stage),
                                        smem_b(cublasdx::slice, cublasdx::slice, stage));
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_STAGE_HPP
