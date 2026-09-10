// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_TMEM_PIPELINE_COMPONENTS_HPP
#define CUBLASDX_DETAIL_PIPELINE_TMEM_PIPELINE_COMPONENTS_HPP

#include "cublasdx/detail/pipeline/tile/rmem/rmem_pipeline_components.hpp"

namespace cublasdx {
    namespace detail {

        using tmem_ptr_t = uint32_t;
        inline constexpr int tmem_alignment_bytes = alignof(tmem_ptr_t);
        inline constexpr int tmem_bytes           = sizeof(tmem_ptr_t);
        inline constexpr int epilogue_named_barrier_id = 1;
        inline constexpr int epilogue_done_named_barrier_id = 3;

        enum class tmem_specialized_warp_role {
            epilogue,
            mma,
            tma
        };

        // Required for EBO - empty base optimization
        template<bool StoreAccumulator, class Accumulator>
        struct tmem_internal_accumulator_storage;

        template<class Accumulator>
        struct tmem_internal_accumulator_storage<true, Accumulator> {
            Accumulator accumulator_;

            CUBLASDX_DEVICE
            explicit tmem_internal_accumulator_storage(Accumulator accumulator): accumulator_(accumulator) {}

            CUBLASDX_DEVICE
            Accumulator& internal_accumulator() { return accumulator_; }

            CUBLASDX_DEVICE
            Accumulator const& internal_accumulator() const { return accumulator_; }
        };

        template<class Accumulator>
        struct tmem_internal_accumulator_storage<false, Accumulator> {
            template<class Empty>
            CUBLASDX_DEVICE
            explicit tmem_internal_accumulator_storage(Empty) {}
        };

        // Execution model for warp-specialized workflow
        template<class BLAS>
        struct tmem_specialized_execution_model {
            static constexpr int epilogue_warps = BLAS::max_threads_per_block / 32;
            static constexpr int mma_warp_idx   = epilogue_warps;
            static constexpr int tma_warp_idx   = epilogue_warps + 1;

            CUBLASDX_DEVICE
            static tmem_specialized_warp_role role(unsigned const warp_idx) {
                if (static_cast<int>(warp_idx) == mma_warp_idx) return tmem_specialized_warp_role::mma;
                if (static_cast<int>(warp_idx) == tma_warp_idx) return tmem_specialized_warp_role::tma;
                return tmem_specialized_warp_role::epilogue;
            }

            CUBLASDX_DEVICE
            static bool is_epilogue(unsigned const warp_idx) {
                // We can assume that warp_idx can be used instead of thread_idx
                // because it's a warp-specialized path: it's launched only in special
                // and divisible cases:
                static_assert(BLAS::max_threads_per_block % 32 == 0);

                return static_cast<int>(warp_idx) < mma_warp_idx;
            }

            CUBLASDX_DEVICE
            static unsigned thread_copy_index(unsigned const tid) {
                constexpr int warp_size = 32;
                return tid - (tma_warp_idx * warp_size);
            }
        };

        // Unified --> all threads perform epilogue, 
        // but 2 are specialized for MMA and TMA respectively
        template<class BLAS>
        struct tmem_unified_execution_model {
            static constexpr int mma_warp_idx = 0;
            static constexpr int tma_warp_idx = 1;

            CUBLASDX_DEVICE
            static bool is_mma(unsigned const warp_idx) {
                return warp_idx == mma_warp_idx;
            }

            CUBLASDX_DEVICE
            static bool is_tma(unsigned const warp_idx) {
                return warp_idx == tma_warp_idx;
            }

            // Relative thread copy index for copies
            // (Only first thread)
            CUBLASDX_DEVICE
            static unsigned thread_copy_index(unsigned const tid) {
                constexpr int warp_size = 32;
                return tid - tma_warp_idx * warp_size;
            }
        };


        // Metainfo for warp-specialized TMEM pipeline
        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 copy_kind      CopyKind,
                 class          BLAS,
                 bool           IsSuggested,
                 int            AdditionalTmaAndMmaThreads,
                 class          PipelinedSmemLayoutA,
                 typename       IOTypeA,
                 class          PipelinedSmemLayoutB,
                 typename       IOTypeB>
        struct tmem_specialized_pipeline_config {
            static constexpr result_storage accumulation = validate_accumulator_mode<ResultStorage>::value;
            static constexpr unsigned sm = sm_of_v<BLAS>;
            // Bytes copied per stage
            static constexpr unsigned copy_bytes =
                (cute::size(PipelinedSmemLayoutA {}) * sizeof(typename BLAS::a_value_type) +
                 cute::size(PipelinedSmemLayoutB {}) * sizeof(typename BLAS::b_value_type)) /
                PipelineDepth;

            // Use traits instead of decltype for avoid device-only function host type inference
            using accumulator_t = cute::conditional_t<IsSuggested,
                typename BLAS::suggested_accumulator_t,
                typename BLAS::default_accumulator_t>;

            // Must be UTCMMA
            static constexpr auto mma_instruction_kind = instruction_type::utcmma;

            static constexpr int additional_tma_and_mma_threads = AdditionalTmaAndMmaThreads;
            static_assert(additional_tma_and_mma_threads == 64,
                          "tmem_specialized_pipeline expects 64 additional TMA/MMA threads");

            // Execution model --> which thread does what
            using execution_model_t = tmem_specialized_execution_model<BLAS>;
            static constexpr int mma_warp_idx       = execution_model_t::mma_warp_idx;
            static constexpr int tma_warp_idx       = execution_model_t::tma_warp_idx;

            // Num epilogue threads --> number designated by user
            static constexpr int num_epilogue_threads = BLAS::max_threads_per_block;
            static constexpr int max_threads_per_block = num_epilogue_threads + additional_tma_and_mma_threads;

            // TMEM stages --> allows to overlap epilogue with MMA/TMA
            static constexpr int accumulator_stages = accumulator_t::accumulator_stages;
            // Allocate ALL columns to infer that address returned will be 0
            static constexpr int tmem_columns = accumulator_t::tmem_columns;
            // 2 barriers for each stages - 1 for reading and 1 for writing
            static constexpr int num_acc_barriers = 2 * accumulator_stages;
            static constexpr unsigned barrier_bytes =
                (2 * PipelineDepth + num_acc_barriers) * sizeof(barrier_storage_t);

            static constexpr int num_copy_smem_barrier_arrivers = 1;
            static constexpr int num_compute_smem_barrier_arrivers = 1;

            using copy_smem_barrier_t =
                decltype(detail::get_copy_smem_barrier<CopyKind, sm, num_copy_smem_barrier_arrivers, copy_bytes>());
            using compute_smem_barrier_t =
                decltype(detail::get_compute_smem_barrier<mma_instruction_kind, sm, num_compute_smem_barrier_arrivers>());
            using epilogue_tmem_barrier_t  = typename accumulator_t::epilogue_tmem_barrier_t;
            using compute_tmem_barrier_t = typename accumulator_t::compute_tmem_barrier_t;
        };

        // Metainfo for unified TMEM pipeline:
        // Warp 0 does MMAs
        // Warp 1 does TMAs
        // All warps perform epilogue
        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 copy_kind      CopyKind,
                 class          BLAS,
                 bool           IsSuggested,
                 int            AdditionalTmaAndMmaThreads,
                 class          PipelinedSmemLayoutA,
                 typename       IOTypeA,
                 class          PipelinedSmemLayoutB,
                 typename       IOTypeB>
        struct tmem_unified_pipeline_config {
            // Ensure external is not used --> Hard deprecation
            static constexpr result_storage accumulation = validate_accumulator_mode<ResultStorage>::value;
            static constexpr unsigned sm = sm_of_v<BLAS>;

            // Bytes copied per stage
            static constexpr unsigned copy_bytes =
                (cute::size(PipelinedSmemLayoutA {}) * sizeof(typename BLAS::a_value_type) +
                 cute::size(PipelinedSmemLayoutB {}) * sizeof(typename BLAS::b_value_type)) /
                PipelineDepth;
            
            // Use traits instead of decltype(function()) to avoid host-device type inference conflicts
            using accumulator_t = cute::conditional_t<IsSuggested,
                typename BLAS::suggested_accumulator_t,
                typename BLAS::default_accumulator_t>;

            // UTCMMA + TMA forced
            static constexpr auto mma_instruction_kind = instruction_type::utcmma;
            static_assert(CopyKind == copy_kind::bulk,
                          "UTCMMA device pipelines require TMA-compatible layouts and bulk TMA copies");

            static constexpr int additional_tma_and_mma_threads = AdditionalTmaAndMmaThreads;
            static_assert(additional_tma_and_mma_threads == 0,
                          "tmem_unified_pipeline does not use additional TMA/MMA threads");

            // Execution model: which warp does what
            using execution_model_t = tmem_unified_execution_model<BLAS>;
            static constexpr int mma_warp_idx = execution_model_t::mma_warp_idx;
            static constexpr int tma_warp_idx = execution_model_t::tma_warp_idx;

            // Epilogue threads: chosen by the user
            // in this case this is the total number of threads
            static constexpr int num_epilogue_threads = BLAS::max_threads_per_block;
            static constexpr int max_threads_per_block = num_epilogue_threads + additional_tma_and_mma_threads;

            // TMEM staging: allows to overlap epilogue and next TMA/MMA
            static constexpr int accumulator_stages = accumulator_t::accumulator_stages;
            // Allocate all columns to allow for assuming tmem_ptr == 0
            static constexpr int tmem_columns = accumulator_t::tmem_columns;
            // 1 barrier for writing 1 barrier for reading
            static constexpr int num_acc_barriers = 2 * accumulator_stages;
            static constexpr unsigned barrier_bytes =
                (2 * PipelineDepth + num_acc_barriers) * sizeof(barrier_storage_t);

            static constexpr int num_copy_smem_barrier_arrivers = 1;
            static constexpr int num_compute_smem_barrier_arrivers = 1;

            using copy_smem_barrier_t =
                decltype(detail::get_copy_smem_barrier<CopyKind, sm, num_copy_smem_barrier_arrivers, copy_bytes>());
            using compute_smem_barrier_t =
                decltype(detail::get_compute_smem_barrier<mma_instruction_kind, sm, num_compute_smem_barrier_arrivers>());
            using epilogue_tmem_barrier_t  = typename accumulator_t::epilogue_tmem_barrier_t;
            using compute_tmem_barrier_t = typename accumulator_t::compute_tmem_barrier_t;
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_TMEM_PIPELINE_COMPONENTS_HPP
