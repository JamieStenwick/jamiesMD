// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_DEVICE_PIPELINE_TRAITS_HPP
#define CUBLASDX_DETAIL_PIPELINE_DEVICE_PIPELINE_TRAITS_HPP

#include "cublasdx/detail/numerical.hpp"
#include "cublasdx/detail/pipeline/tile/tmem/tmem_pipeline_components.hpp"

namespace cublasdx {
    namespace detail {

        // Use traits instead of decltype(BLAS().suggest...()) to make 
        // host trait querying safe (device functions are void to host)
        template<class BLAS, bool IsSuggested>
        using selected_accumulator_t = cute::conditional_t<IsSuggested,
            typename BLAS::suggested_accumulator_t,
            typename BLAS::default_accumulator_t>;

        enum class pipeline_family {
            rmem_unified,
            rmem_specialized,
            tmem_unified,
            tmem_specialized
        };

        // Unified --> non warp-specialized
        struct rmem_unified_pipeline_family_traits {
            static constexpr pipeline_family family = pipeline_family::rmem_unified;
            static constexpr int additional_tma_and_mma_threads = 0;

            template<class BLAS>
            static constexpr dim3 block_dim() {
                return BLAS::block_dim;
            }

            template<unsigned PipelineDepth, class BLAS, bool IsSuggested = false>
            static constexpr unsigned barrier_bytes() {
                return 2 * PipelineDepth * sizeof(barrier_storage_t);
            }

            static constexpr unsigned shared_storage_size(unsigned shared_bytes_base) {
                return shared_bytes_base;
            }
        };

        // Specialized --> Warp specialized, extra warps for TMA
        // Epilogue remains coupled with MMA
        struct rmem_specialized_pipeline_family_traits {
            static constexpr pipeline_family family = pipeline_family::rmem_specialized;
            static constexpr int additional_tma_and_mma_threads = 128;

            template<class BLAS>
            static constexpr dim3 block_dim() {
                return dim3(BLAS::max_threads_per_block + additional_tma_and_mma_threads, 1, 1);
            }

            template<unsigned PipelineDepth, class BLAS, bool IsSuggested = false>
            static constexpr unsigned barrier_bytes() {
                return 2 * PipelineDepth * sizeof(barrier_storage_t);
            }

            static constexpr unsigned shared_storage_size(unsigned shared_bytes_base) {
                return shared_bytes_base;
            }
        };

        // TMEM Specialized --> 3 groups, 1 extra warp for MMA and 1 extra warp for TMA
        // Epilogue is now decoupled from MMA and happens on independent TMEM buffers
        struct tmem_specialized_pipeline_family_traits {
            static constexpr pipeline_family family = pipeline_family::tmem_specialized;
            static constexpr int additional_tma_and_mma_threads = 64;

            template<class BLAS>
            static constexpr dim3 block_dim() {
                return dim3(BLAS::max_threads_per_block + additional_tma_and_mma_threads, 1, 1);
            }

            template<unsigned PipelineDepth, class BLAS, bool IsSuggested = false>
            static constexpr int num_acc_barriers() {
                return 2 * selected_accumulator_t<BLAS, IsSuggested>::accumulator_stages;
            }

            template<unsigned PipelineDepth, class BLAS, bool IsSuggested = false>
            static constexpr unsigned barrier_bytes() {
                return (2 * PipelineDepth + num_acc_barriers<PipelineDepth, BLAS, IsSuggested>()) * sizeof(barrier_storage_t);
            }

            static constexpr unsigned shared_storage_size(unsigned shared_bytes_base) {
                return round_up(shared_bytes_base, tmem_alignment_bytes) + tmem_bytes;
            }
        };

        // TMEM but without WS
        struct tmem_unified_pipeline_family_traits {
            static constexpr pipeline_family family = pipeline_family::tmem_unified;
            static constexpr int additional_tma_and_mma_threads = 0;

            template<class BLAS>
            static constexpr dim3 block_dim() {
                return BLAS::block_dim;
            }

            template<unsigned PipelineDepth, class BLAS, bool IsSuggested = false>
            static constexpr int num_acc_barriers() {
                return 2 * selected_accumulator_t<BLAS, IsSuggested>::accumulator_stages;
            }

            template<unsigned PipelineDepth, class BLAS, bool IsSuggested = false>
            static constexpr unsigned barrier_bytes() {
                return (2 * PipelineDepth + num_acc_barriers<PipelineDepth, BLAS, IsSuggested>()) * sizeof(barrier_storage_t);
            }

            static constexpr unsigned shared_storage_size(unsigned shared_bytes_base) {
                return round_up(shared_bytes_base, tmem_alignment_bytes) + tmem_bytes;
            }
        };

        template<instruction_type InstructionKind, bool IsWarpSpecialized, bool IsTma>
        struct pipeline_family_selector {
            using type = rmem_unified_pipeline_family_traits;
        };

        template<>
        struct pipeline_family_selector<instruction_type::gmma, true, true> {
            using type = rmem_specialized_pipeline_family_traits;
        };

        template<>
        struct pipeline_family_selector<instruction_type::utcmma, false, true> {
            using type = tmem_unified_pipeline_family_traits;
        };

        template<>
        struct pipeline_family_selector<instruction_type::utcmma, true, true> {
            using type = tmem_specialized_pipeline_family_traits;
        };

        // Extracted for readability
        template<unsigned       PipelineDepth,
                 bool           DisableWarpSpecialization,
                 copy_kind      CopyKind,
                 class          BLAS,
                 bool           IsSuggested>
        struct device_pipeline_traits {
            static constexpr auto is_tma         = cute::C<CopyKind == copy_kind::bulk> {};

            static constexpr auto mma_instruction_kind =
                IsSuggested ? BLAS::swizzled_instruction_kind : BLAS::default_instruction_kind;
            static constexpr bool can_warp_specialize =
                // Limit WS only to GMMA and UTCMMA --> this is an assumption in superMMA code at the moment
                // if lifted remember to change accumulator.is_thread_active() as well 
                (mma_instruction_kind == instruction_type::utcmma or mma_instruction_kind == instruction_type::gmma) and
                // Only 90a and 100a, 101a, 103a can specialize and only if threads are below 256
                // This is a heuristic more than HW forced rule for GMMA
                (sm_of<BLAS>::modifier == sm_modifier::arch_specific) and (BLAS::max_threads_per_block <= 256) and
                // For simplicity of calculations require all threads to be in DimX for WS
                (BLAS::block_dim.y == 1 and BLAS::block_dim.z == 1) and 
                // Sanity check for special cases and require TMA for WS
                (not DisableWarpSpecialization) and is_tma;
            static_assert(mma_instruction_kind != instruction_type::utcmma or is_tma,
                          "UTCMMA device pipelines require TMA-compatible layouts and bulk TMA copies");

            using pipeline_family_t =
                typename pipeline_family_selector<mma_instruction_kind, can_warp_specialize, is_tma>::type;

            static constexpr bool is_warp_specialized =
                pipeline_family_t::family == pipeline_family::rmem_specialized or
                pipeline_family_t::family == pipeline_family::tmem_specialized;
            static constexpr bool is_tmem_unified =
                pipeline_family_t::family == pipeline_family::tmem_unified;
            static constexpr bool is_tmem_specialized =
                pipeline_family_t::family == pipeline_family::tmem_specialized;
            static constexpr bool is_rmem_specialized =
                pipeline_family_t::family == pipeline_family::rmem_specialized;

            // Threads on top of users' BlockDim for async instruction dispatch
            static constexpr int additional_tma_and_mma_threads =
                pipeline_family_t::additional_tma_and_mma_threads;
            static_assert(not is_tmem_specialized or additional_tma_and_mma_threads == 64,
                          "tmem_specialized pipelines use 64 additional TMA/MMA threads");
            static_assert(not is_rmem_specialized or additional_tma_and_mma_threads == 128,
                          "rmem_specialized pipelines use 128 additional TMA threads");
            // Accessible for launch bounds
            static constexpr int max_threads_per_block = BLAS::max_threads_per_block + additional_tma_and_mma_threads;
            // First warp outside users' BlockDim. RMEM uses it as the first TMA warp.
            static constexpr int first_extra_warp_idx =
                is_warp_specialized ? (BLAS::max_threads_per_block / 32) : 0;

            static constexpr unsigned alignment_a = is_tma ? tma_buffer_alignment_bytes : alignment_of_v_a<BLAS>;
            static constexpr unsigned alignment_b = is_tma ? tma_buffer_alignment_bytes : alignment_of_v_b<BLAS>;

            static constexpr unsigned barrier_bytes =
                pipeline_family_t::template barrier_bytes<PipelineDepth, BLAS, IsSuggested>();

            static constexpr dim3 block_dim = pipeline_family_t::template block_dim<BLAS>();

            static constexpr unsigned shared_storage_size(unsigned shared_bytes_base) {
                return pipeline_family_t::shared_storage_size(shared_bytes_base);
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_DEVICE_PIPELINE_TRAITS_HPP
