// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_RMEM_SPECIALIZED_PIPELINE_HPP
#define CUBLASDX_DETAIL_PIPELINE_RMEM_SPECIALIZED_PIPELINE_HPP

#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/barrier.h>

#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/detail/copy.hpp"
#include "cublasdx/detail/shared_memory.hpp"

#include "cublasdx/detail/pipeline/barriers/barrier.hpp"
#include "cublasdx/detail/pipeline/logging.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_base.hpp"
#include "cublasdx/detail/pipeline/tile/rmem/rmem_pipeline_components.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_traits.hpp"

namespace cublasdx {
    namespace detail {

        // RMEM specialized pipeline performs Hopper style copy/compute warp-specialization
        // where user requested BlockDim does MMA and epilogue and a decoupled, independent warpgroup
        // does TMA loads. 
        // Register trading is possible
        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 copy_kind      CopyKind,
                 class BLAS,
                 bool IsSuggested,
                 int  NumCopyThreads,
                 int  FirstCopyWarpIdx,
                 class GmemDescriptorA,
                 class CopyFunctorA,
                 class GmemDescriptorB,
                 class CopyFunctorB,
                 class PipelinedSmemLayoutA,
                 typename IOTypeA,
                 class PipelinedSmemLayoutB,
                 typename IOTypeB>
        struct rmem_specialized_pipeline :
            public tile_pipeline_base<PipelineDepth, ResultStorage, BLAS,
                                 GmemDescriptorA, CopyFunctorA, GmemDescriptorB, CopyFunctorB,
                                 PipelinedSmemLayoutA, IOTypeA, PipelinedSmemLayoutB, IOTypeB> {

            using base_type = tile_pipeline_base<PipelineDepth, ResultStorage, BLAS,
                                            GmemDescriptorA, CopyFunctorA, GmemDescriptorB, CopyFunctorB,
                                            PipelinedSmemLayoutA, IOTypeA, PipelinedSmemLayoutB, IOTypeB>;
            using base_type::tid;
            using base_type::warp_idx;
            using base_type::read_cursor;
            using base_type::write_cursor;
            using base_type::k_chunks;
            using base_type::gmem_a;
            using base_type::gmem_b;
            using base_type::copy_functor_a;
            using base_type::copy_functor_b;
            using base_type::shared_tensor_storage;
            using base_type::accumulation;
            static constexpr unsigned sm               = base_type::sm;
            static constexpr bool     has_static_block_dim = base_type::has_static_block_dim;
            static constexpr int      rank             = base_type::rank;

            using config = rmem_pipeline_config<PipelineDepth,
                                                ResultStorage,
                                                CopyKind,
                                                BLAS,
                                                IsSuggested,
                                                NumCopyThreads,
                                                FirstCopyWarpIdx,
                                                PipelinedSmemLayoutA,
                                                IOTypeA,
                                                PipelinedSmemLayoutB,
                                                IOTypeB>;
            using execution_model_t = typename config::execution_model_t;

            static constexpr auto is_warp_specialized = config::is_warp_specialized;
            static constexpr auto is_cooperative      = config::is_cooperative;
            static constexpr auto is_tma              = config::is_tma;

            static_assert(NumCopyThreads == 128,
                          "rmem_specialized_pipeline expects 128 additional TMA threads");
            static_assert(is_warp_specialized, "Use rmem_unified_pipeline for non-warp-specialized execution");
            static_assert(not is_warp_specialized or is_tma);
            static_assert(not(is_warp_specialized and is_cooperative));

            static constexpr auto mma_instruction_kind = config::mma_instruction_kind;
            // Only GMMA or superMMA can take part
            static constexpr auto is_gmma   = config::is_gmma;

            static constexpr unsigned barrier_bytes = config::barrier_bytes;

            static_assert(mma_instruction_kind != instruction_type::utcmma,
                          "UTCMMA pipelines must use tmem_unified_pipeline or tmem_specialized_pipeline");

            // Only 1 thread issues TMA
            bool const first_copy_predicate    = cute::elect_one_sync() and (warp_idx == FirstCopyWarpIdx);

            // TMA threads
            static constexpr int num_copy_threads = config::num_copy_threads;
            // BlockDim
            static constexpr int num_compute_threads = config::num_compute_threads;
            static constexpr int max_threads_per_block = config::max_threads_per_block;

            using copy_smem_barrier_t = typename config::copy_smem_barrier_t;
            using compute_smem_barrier_t = typename config::compute_smem_barrier_t;

            barrier_group<copy_smem_barrier_t, PipelineDepth> copy_smem_barriers {};
            barrier_group<compute_smem_barrier_t, PipelineDepth> compute_smem_barriers {};

            CUBLASDX_DEVICE
            bool is_copy_warp() const {
                return execution_model_t::is_copy_warp(warp_idx);
            }

            CUBLASDX_DEVICE
            bool is_compute_warp() const {
                return execution_model_t::is_compute_warp(warp_idx);
            }

            // Used for gating by BLAS::execute_pipeline
            CUBLASDX_DEVICE
            bool should_issue_mma() const {
                return is_compute_warp();
            }

            CUBLASDX_DEVICE
            bool should_issue_copy() const {
                return (is_cooperative and is_copy_warp()) or (is_tma and first_copy_predicate);
            }

            // Lifetime methods - IMPLICITLY PUBLIC, NEVER CALLED DIRECTLY
            CUBLASDX_DEVICE
            rmem_specialized_pipeline(int                         count,
                                       GmemDescriptorA const&      in_gmem_a,
                                       CopyFunctorA                copy_a,
                                       GmemDescriptorB const&      in_gmem_b,
                                       CopyFunctorB                copy_b,
                                       byte*                       smem,
                                       PipelinedSmemLayoutA const& layout_a,
                                       PipelinedSmemLayoutB const& layout_b):
                base_type(count, in_gmem_a, copy_a, in_gmem_b, copy_b) {

                constexpr unsigned a_alignment = is_tma ? tma_buffer_alignment_bytes : alignment_of_v_a<BLAS>;
                constexpr unsigned b_alignment = is_tma ? tma_buffer_alignment_bytes : alignment_of_v_b<BLAS>;

                static_assert(barrier_bytes % barrier_buffer_alignment_bytes == 0);

                auto [sliced_smem_a, sliced_smem_b, barriers] =
                    shared_memory::slice<IOTypeA, IOTypeB, barrier_storage_t>(
                        smem,
                        a_alignment,
                        layout_a,
                        b_alignment,
                        layout_b,
                        barrier_buffer_alignment_bytes,
                        barrier_bytes / barrier_buffer_alignment_bytes);

                copy_smem_barriers.reset(barriers);
                compute_smem_barriers.reset(barriers + PipelineDepth);
                shared_tensor_storage.reset(sliced_smem_a, sliced_smem_b);

                // Only 1 thread initializes barriers
                if (first_copy_predicate) {
                    copy_smem_barriers.init_all();
                    compute_smem_barriers.init_all();
                }

                // make mbarrier state visible for everyone in this CTA
                __syncthreads();
            }

            // INTERNAL -- IMPLEMENTATION

            // Wait till TMA has finished filling buffer stage and it's ready to be multiplied with MMA
            CUBLASDX_DEVICE
            void compute_smem_acquire() {
                PIPELINE_LOG(
                    "compute_smem_acquire tid %d stage %d phase %d\n", tid, read_cursor.index(), read_cursor.phase());
                // Wait until all copy threads have writter all data into this buffer
                if (is_compute_warp()) {
                    copy_smem_barriers[read_cursor.index()].wait(read_cursor.phase());
                }

                if constexpr (is_gmma and not is_tma) {
                    asm volatile("fence.proxy.async.shared::cta; \n" ::: "memory");
                }
            };

            // Signal that buffer stage is free to be stored into again
            CUBLASDX_DEVICE
            void compute_smem_commit() {
                PIPELINE_LOG("compute_smem_commit tid %d stage %d \n", tid, read_cursor.index());

                unsigned const pipe_read = read_cursor.index();

                if constexpr (sm_of_v<BLAS> >= 900 and is_tma) {
                    asm volatile("fence.proxy.async.shared::cta; \n" ::: "memory");
                } else {
                    asm volatile("membar.cta; \n" ::: "memory");
                }

                if (is_compute_warp()) {
                    compute_smem_barriers[pipe_read].arrive_after();
                }

                read_cursor.advance();
            };

            private:

            // Wait till MMA warps finish operations on shared memory buffer and it's ready to be loaded into
            CUBLASDX_DEVICE
            void copy_smem_acquire() {
                PIPELINE_LOG(
                    "copy_smem_acquire tid %d stage %d phase %d\n", tid, write_cursor.index(), write_cursor.phase());

                unsigned const pipe_write = write_cursor.index();
                if (is_copy_warp()) {
                    compute_smem_barriers[pipe_write].wait(write_cursor.phase());
                }
                if (should_issue_copy()) {
                    copy_smem_barriers[pipe_write].arrive_before();
                }
            };

            // Signal that TMA is done loading data into a shared memory buffer
            CUBLASDX_DEVICE
            void copy_smem_commit() {
                PIPELINE_LOG("copy_smem_commit tid %d stage %d\n", tid, write_cursor.index());

                if (is_copy_warp()) {
                    copy_smem_barriers[write_cursor.index()].arrive_after();
                }
                write_cursor.advance();
            };

            // Perform shared memory load from GlobalStage into SharedStage
            template<class GlobalStage, class SharedStage>
            CUBLASDX_DEVICE void schedule_current_ab_load(GlobalStage global_stage, SharedStage shared_stage) {
                PIPELINE_LOG("Scheduling A/B tid %d chunk %d to stage %d\n",
                             tid,
                             unsigned(global_stage),
                             unsigned(shared_stage));

                const auto copy_tid = execution_model_t::thread_copy_index(tid);

                auto write_barrier = copy_smem_barriers[shared_stage];

                // Depending on rank of global tensor, the k-stage is placed either
                // in the 2nd or 3rd index
                auto const global_coord = cute::conditional_return<rank == 2>(
                    cute::make_coord(cublasdx::slice, cublasdx::slice, global_stage),
                    cute::make_coord(cublasdx::slice, cublasdx::slice, 0, global_stage));

                copy_functor_a(copy_tid,
                               write_barrier,
                               gmem_a(global_coord),
                               shared_tensor_storage.smem_a(cublasdx::slice, cublasdx::slice, shared_stage));

                copy_functor_b(copy_tid,
                               write_barrier,
                               gmem_b(global_coord),
                               shared_tensor_storage.smem_b(cublasdx::slice, cublasdx::slice, shared_stage));
            }

            // Batch schedule initial <Stages> TMA loads so that as much data is in flight as possible
            // This step guarantees N-1 stages in flight while single stage is being computed on
            CUBLASDX_DEVICE
            void prefill() {
                PIPELINE_LOG("Copying A/B tid %d chunk %d to stage %d\n", tid, write_cursor.chunk, write_cursor.index());

                // Static automatically unrolled loop
                cute::for_each(cute::make_int_sequence<PipelineDepth> {}, [&](auto stage) {
                    copy_smem_acquire();
                    if (should_issue_copy()) {
                        schedule_current_ab_load(stage, write_cursor.index());
                    }
                    copy_smem_commit();
                });
            };

            // Acquire Stage for load -> Gated Copy -> Release stage for compute
            CUBLASDX_DEVICE
            void copy_next() {
                PIPELINE_LOG("Copying A/B tid %d chunk %d to stage %d\n", tid, write_cursor.chunk, write_cursor.index());

                copy_smem_acquire();

                if (should_issue_copy()) {
                    schedule_current_ab_load(write_cursor.chunk, write_cursor.index());
                }

                copy_smem_commit();
            };

            // Mainloop for TMA warps
            CUBLASDX_DEVICE
            void execute_copy_warp_loop() {
                static_assert(is_warp_specialized);
                assert(is_copy_warp());
                // If TMA then only first warp can enter
                if (not is_tma or warp_idx == FirstCopyWarpIdx) {
                    this->prefill();
                    // Mainloop condition
                    while (this->has_copy_smem_work()) {
                        // Schedule next copy
                        this->copy_next();
                        PIPELINE_LOG("copy tid: %d stage: %d\n", threadIdx.x, copy_smem_chunk());
                    }
                }
            }

            // Mainloop for compute warps
            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE void execute_compute_warp_loop(BlasAccumulator&& accumulator,
                                                  ALoadOp const&    a_load_op = {},
                                                  BLoadOp const&    b_load_op = {}) {
                static_assert(is_warp_specialized);
                assert(is_compute_warp());

                // Mainloop condition
                while (this->has_compute_smem_work()) {
                    PIPELINE_LOG("compute tid: %d stage: %d\n", threadIdx.x, compute_smem_chunk());
                    // Internal call to avoid executes -> WithPipeline disables all execute()
                    BLAS().execute_from_pipeline(*this, accumulator, a_load_op, b_load_op);
                }
            }

            // Register trading between copy and compute warp
            template<bool IsProducer>
            CUBLASDX_DEVICE void attempt_register_trading() {
	        static_assert(is_warp_specialized, "This method is available only for warp specialized pipelines");

                static constexpr int registers_per_copy_thread = 24;

                static constexpr int total_registers = 64 * 1024;
                static constexpr int total_compute_registers = (size_of_v_m<BLAS> * size_of_v_n<BLAS>) * sizeof(float) / sizeof(typename BLAS::c_value_type);
                static constexpr int required_registers_per_compute_thread = total_compute_registers / num_compute_threads;

                static constexpr int total_copy_registers  = num_copy_threads * registers_per_copy_thread;
                // Round down to nearest multiple of 8
                static constexpr int registers_per_compute_thread = 8 * (((total_registers - total_copy_registers) / num_compute_threads) / 8);

                // If ThreadCount <= 255 then each thread already owns 255 regs and nothing can be improved
                if constexpr (registers_per_compute_thread >= required_registers_per_compute_thread and
                              accumulation == internal_accumulation and
                              max_threads_per_block > 256 and
                              sm == 900) {
                    if constexpr(IsProducer) {
                        cutlass::detail::NamedBarrierSync<num_copy_threads, copy_smem_named_barrier_id>::sync();
                        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(registers_per_copy_thread));
                    } else {
                        cutlass::detail::NamedBarrierSync<num_compute_threads, compute_smem_named_barrier_id>::sync();
                        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(registers_per_compute_thread));
                    }
                }
            }

            // Execute with external accumulator
            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value>
                            execute_warp_specialized(BlasAccumulator&& accumulator,
                                                     ALoadOp const&    a_load_op = {},
                                                     BLoadOp const&    b_load_op = {}) {
                static_assert(is_warp_specialized);
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");

                if (is_copy_warp()) {
                    attempt_register_trading<true>();
                    execute_copy_warp_loop();
                }

                if (is_compute_warp()) {
                    attempt_register_trading<false>();
                    execute_compute_warp_loop(accumulator, a_load_op, b_load_op);
                }
            }

            // Execute with internal accumulator
            template<class EpilogueFunctor, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<not is_valid_blas_accumulator<EpilogueFunctor>::value>
                            execute_warp_specialized(EpilogueFunctor const& epilogue_functor,
                                                     ALoadOp const&         a_load_op = {},
                                                     BLoadOp const&         b_load_op = {}) {
                static_assert(is_warp_specialized);
                static_assert(accumulation == internal_accumulation, "This method is available only for internal accumulation");

                if (is_copy_warp()) {
                    attempt_register_trading<true>();
                    execute_copy_warp_loop();
                }

                if (is_compute_warp()) {
                    attempt_register_trading<false>();
                    // it's important that accumulator is created here after IF
                    // because register trading may be used and then only compute
                    // threads should own more RF
                    auto accumulator = get_accumulator_impl();
                    execute_compute_warp_loop(accumulator, a_load_op, b_load_op);
                    if (accumulator.is_thread_active()) {
                        epilogue_functor(accumulator);
                    }
                }
            }

            CUBLASDX_DEVICE
            auto get_accumulator_impl() {
                if constexpr (IsSuggested) {
                    return BLAS().suggest_accumulator_impl();
                } else {
                    return BLAS().get_accumulator_impl();
                }
            }

            public:

            CUBLASDX_DEVICE
            auto get_accumulator() {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");
                return get_accumulator_impl();
            }

            // This can be used inside epilogue functors
            CUBLASDX_DEVICE
            void epilogue_sync() {
                if (is_compute_warp()) {
                    cutlass::detail::NamedBarrierSync<num_compute_threads, compute_smem_named_barrier_id>::sync();
                }
            }

            // Dispatching execute for external accumulator
            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator_v<BlasAccumulator>>
            execute(BlasAccumulator&& accumulator, ALoadOp const& a_load_op = {}, BLoadOp const& b_load_op = {}) {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");
                execute_warp_specialized(accumulator, a_load_op, b_load_op);
            }

            // General epilogue (provide gating) for external accumulators
            // Importantly it provides automatic gating
            template<class BlasAccumulator, class EpilogueFunctor>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value>
            epilogue(BlasAccumulator&& accumulator,
                     EpilogueFunctor const& epilogue_functor) {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");
                accumulator.finish_accumulation();
                if (is_compute_warp() && accumulator.is_thread_active()) {
                    epilogue_functor(accumulator);
                }
            }

            // General execute for internal accumulator
            template<class EpilogueFunctor, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<not is_valid_blas_accumulator_v<EpilogueFunctor>>
            execute(EpilogueFunctor&& epilogue_functor, ALoadOp const& a_load_op = {}, BLoadOp const& b_load_op = {}) {
                static_assert(accumulation == internal_accumulation, "This method is available only for internal accumulation");
                execute_warp_specialized(epilogue_functor, a_load_op, b_load_op);
            }
        };

        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 copy_kind      CopyKind,
                 class BLAS,
                 bool IsSuggested,
                 int  NumCopyThreads,
                 int  FirstCopyWarpIdx,
                 class... Rest>
        struct is_pipeline<rmem_specialized_pipeline<PipelineDepth,
                                                      ResultStorage,
                                                      CopyKind,
                                                      BLAS,
                                                      IsSuggested,
                                                      NumCopyThreads,
                                                      FirstCopyWarpIdx,
                                                      Rest...>>: cute::true_type {
        };

    } // namespace detail
} // namespace cublasdx
#endif // CUBLASDX_DETAIL_PIPELINE_RMEM_SPECIALIZED_PIPELINE_HPP
