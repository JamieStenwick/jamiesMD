// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_RMEM_PIPELINE_COMPONENTS_HPP
#define CUBLASDX_DETAIL_PIPELINE_RMEM_PIPELINE_COMPONENTS_HPP

#include "cublasdx/detail/pipeline/barriers/barrier.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_stage.hpp"

namespace cublasdx {
    namespace detail {

        inline constexpr int compute_smem_named_barrier_id = 1;
        inline constexpr int copy_smem_named_barrier_id = 2;

        template<bool IsWarpSpecialized, int FirstCopyWarpIdx>
        struct rmem_pipeline_execution_model {
            // Is this warp responsible for TMA
            CUBLASDX_DEVICE
            static bool is_copy_warp(unsigned const warp_idx) {
                return !IsWarpSpecialized ||
                       (static_cast<int>(warp_idx) >= static_cast<int>(FirstCopyWarpIdx));
            }

            // Is this warp responsible for MMA/Epilogue
            CUBLASDX_DEVICE
            static bool is_compute_warp(unsigned const warp_idx) {
                return !IsWarpSpecialized ||
                       (static_cast<int>(warp_idx) < static_cast<int>(FirstCopyWarpIdx));
            }

            // For warp-specialized find threadIdx inside local 
            // group specialized to copying only
            CUBLASDX_DEVICE
            static unsigned thread_copy_index(unsigned const tid) {
                constexpr int warp_size = 32;
                if constexpr (FirstCopyWarpIdx > 0) {
                    assert(tid >= FirstCopyWarpIdx * warp_size);
                }
                return tid - FirstCopyWarpIdx * warp_size;
            }
        };

        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 copy_kind      CopyKind,
                 class          BLAS,
                 bool           IsSuggested,
                 int            NumCopyThreads,
                 int            FirstCopyWarpIdx,
                 class          PipelinedSmemLayoutA,
                 typename       IOTypeA,
                 class          PipelinedSmemLayoutB,
                 typename       IOTypeB>
        struct rmem_pipeline_config {
            // Check if external_accumulation is not used, just a static assert
            static constexpr result_storage accumulation = validate_accumulator_mode<ResultStorage>::value;
            static constexpr unsigned sm = sm_of_v<BLAS>;
            // Per stage
            static constexpr unsigned copy_bytes =
                (cute::size(PipelinedSmemLayoutA {}) * sizeof(typename BLAS::a_value_type) +
                 cute::size(PipelinedSmemLayoutB {}) * sizeof(typename BLAS::b_value_type)) /
                PipelineDepth;
            static constexpr bool has_static_block_dim = cublasdx::has_static_block_dim_v<BLAS>;

            static constexpr auto is_warp_specialized = cute::C<(FirstCopyWarpIdx != 0)> {};
            // Is cute::cooperative_copy used (LDGSTS or LDG+STS)
            static constexpr auto is_cooperative      = cute::C<CopyKind != copy_kind::bulk> {};
            static constexpr auto is_tma              = cute::C<CopyKind == copy_kind::bulk> {};

            static constexpr auto mma_instruction_kind =
                IsSuggested ? BLAS::swizzled_instruction_kind : BLAS::default_instruction_kind;
            // RMEM can be only either GMMA or superMMA
            static constexpr auto is_gmma = cute::C<mma_instruction_kind == instruction_type::gmma> {};

            static constexpr unsigned barrier_bytes = 2 * PipelineDepth * sizeof(barrier_storage_t);

            static constexpr int num_copy_threads = NumCopyThreads;
            static constexpr int num_compute_threads = BLAS::max_threads_per_block;
            static constexpr int max_threads_per_block =
                is_warp_specialized ? (num_copy_threads + num_compute_threads) : num_compute_threads;

            static constexpr int num_copy_smem_barrier_arrivers = is_tma ? 1 : NumCopyThreads;
            static constexpr int num_compute_smem_barrier_arrivers = num_compute_threads;

            using copy_smem_barrier_t =
                decltype(detail::get_copy_smem_barrier<CopyKind, sm, num_copy_smem_barrier_arrivers, copy_bytes>());
            using compute_smem_barrier_t =
                decltype(detail::get_compute_smem_barrier<mma_instruction_kind, sm, num_compute_smem_barrier_arrivers>());

            using execution_model_t =
                rmem_pipeline_execution_model<static_cast<bool>(is_warp_specialized), FirstCopyWarpIdx>;
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_RMEM_PIPELINE_COMPONENTS_HPP
