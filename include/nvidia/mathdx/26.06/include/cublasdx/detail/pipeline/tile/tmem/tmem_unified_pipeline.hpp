// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_TMEM_UNIFIED_PIPELINE_HPP
#define CUBLASDX_DETAIL_PIPELINE_TMEM_UNIFIED_PIPELINE_HPP

#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/barrier.h>

#include "commondx/detail/stl/type_traits.hpp"

#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/detail/copy.hpp"
#include "cublasdx/detail/shared_memory.hpp"

#include "cublasdx/detail/pipeline/accumulator_mode.hpp"
#include "cublasdx/detail/pipeline/logging.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_base.hpp"
#include "cublasdx/detail/pipeline/tile/tmem/tmem_pipeline_components.hpp"
#include "cublasdx/detail/pipeline/tile/rmem/rmem_pipeline_components.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_traits.hpp"

namespace cublasdx {
    namespace detail {

        // TMEM based (Blackwell 100/110/103) unified (non-warp-specialized) pipeline
        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 copy_kind      CopyKind,
                 class BLAS,
                 bool IsSuggested,
                 int  AdditionalTmaAndMmaThreads,
                 class GmemDescriptorA,
                 class CopyFunctorA,
                 class GmemDescriptorB,
                 class CopyFunctorB,
                 class PipelinedSmemLayoutA,
                 typename IOTypeA,
                 class PipelinedSmemLayoutB,
                 typename IOTypeB>
        struct tmem_unified_pipeline :
            public tile_pipeline_base<PipelineDepth, ResultStorage, BLAS,
                                 GmemDescriptorA, CopyFunctorA, GmemDescriptorB, CopyFunctorB,
                                 PipelinedSmemLayoutA, IOTypeA, PipelinedSmemLayoutB, IOTypeB>,
            private tmem_internal_accumulator_storage<
                validate_accumulator_mode<ResultStorage>::value == internal_accumulation,
                cute::conditional_t<IsSuggested, typename BLAS::suggested_accumulator_t, typename BLAS::default_accumulator_t>> {

            static_assert(AdditionalTmaAndMmaThreads == 0,
                          "tmem_unified_pipeline does not use additional TMA/MMA threads");

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
            using base_type::has_static_block_dim;

            static constexpr int      rank = base_type::rank;

            using config = tmem_unified_pipeline_config<PipelineDepth,
                                                         ResultStorage,
                                                         CopyKind,
                                                         BLAS,
                                                         IsSuggested,
                                                         AdditionalTmaAndMmaThreads,
                                                         PipelinedSmemLayoutA,
                                                         IOTypeA,
                                                         PipelinedSmemLayoutB,
                                                         IOTypeB>;
            using execution_model_t = typename config::execution_model_t;

            // Supported only for UTCMMA
            static constexpr auto mma_instruction_kind = config::mma_instruction_kind;

            static_assert(mma_instruction_kind == instruction_type::utcmma,
                          "tmem_unified_pipeline is only for UTCMMA instructions");
            static_assert(CopyKind == copy_kind::bulk, "UTCMMA should always be paired with TMA");

            static constexpr unsigned barrier_bytes = config::barrier_bytes;

            // This pipeline, although unified, will need to execute many instructions
            // with only some threads. e.g. TMA and UTCMMA launch require only a single
            // thread
            static constexpr int mma_warp_idx       = config::mma_warp_idx;
            static constexpr int tma_warp_idx       = config::tma_warp_idx;
            static_assert(mma_warp_idx != tma_warp_idx, "Designate 2 different warps for MMA/TMA");

            // Designate a single thread for doing MMA and TMA
            bool const first_copy_predicate    =
                cute::elect_one_sync() and (warp_idx == static_cast<unsigned>(tma_warp_idx));
            bool const first_compute_predicate =
                cute::elect_one_sync() and (warp_idx == static_cast<unsigned>(mma_warp_idx));

            static constexpr int additional_tma_and_mma_threads = config::additional_tma_and_mma_threads;
            static constexpr int max_threads_per_block = config::max_threads_per_block;

            // Will be equal to total number of threads - user requested is launched
            static constexpr int num_epilogue_threads = config::num_epilogue_threads;

            // TMEM accumulator stages for independent tile execution
            static constexpr int accumulator_stages = config::accumulator_stages;
            static constexpr int tmem_columns = config::tmem_columns;

            // Owned by pipeline
            using copy_smem_barrier_t = typename config::copy_smem_barrier_t;
            using compute_smem_barrier_t = typename config::compute_smem_barrier_t;

            // Owned by accumulator, initialized by pipeline
            using epilogue_tmem_barrier_t = typename config::epilogue_tmem_barrier_t;
            using compute_tmem_barrier_t = typename config::compute_tmem_barrier_t;

            // Owned by pipeline
            barrier_group<copy_smem_barrier_t, PipelineDepth> copy_smem_barriers {};
            barrier_group<compute_smem_barrier_t, PipelineDepth> compute_smem_barriers {};
            // Passed to accumulator later
            barrier_group<compute_tmem_barrier_t, accumulator_stages> compute_tmem_barriers {};
            barrier_group<epilogue_tmem_barrier_t, accumulator_stages> epilogue_tmem_barriers {};

            using accumulator_t = typename config::accumulator_t;
            using internal_accumulator_base =
                tmem_internal_accumulator_storage<accumulation == internal_accumulation, accumulator_t>;

            // For internal accumulator
            CUBLASDX_DEVICE
            static accumulator_t make_accumulator() {
                if constexpr (IsSuggested) {
                    return BLAS().suggest_accumulator_impl();
                } else {
                    return BLAS().get_accumulator_impl();
                }
            }

            // Used for EBO based storage of internal accumulator
            // Empty Base Optimization: this class is derived from accumulator_owner class, and if 
            // that class is empty then there will be no extra storage cost incurred. 
            // If the internal accumulator member class would be directly contained then 
            // even in case of using a temporary empty structure (like tuple<>) a cost of 1-byte 
            // is always assumed.
            using internal_accumulator_t = cute::conditional_t<accumulation == internal_accumulation,
                                                               accumulator_t,
                                                               cute::tuple<>>;

            CUBLASDX_DEVICE
            static internal_accumulator_t make_internal_accumulator() {
                if constexpr (accumulation == internal_accumulation) {
                    return make_accumulator();
                } else {
                    return {};
                }
            }

            CUBLASDX_DEVICE
            bool is_active() const {
                return tid < num_epilogue_threads;
            }

            // Special warp for MMA issuing
            CUBLASDX_DEVICE
            bool is_mma() const {
                return execution_model_t::is_mma(warp_idx);
            }

            // Special warp for TMA issuing
            CUBLASDX_DEVICE
            bool is_tma() const {
                return execution_model_t::is_tma(warp_idx);
            }

            CUBLASDX_DEVICE
            bool should_issue_tma() const {
                return first_copy_predicate;
            }

            // TID based --> elect based, 
            // should not be used for large chunks for code
            // cute::gemm calls elect() on MMA internally
            CUBLASDX_DEVICE
            bool should_issue_mma() const {
                return is_mma();
            }

            // Use Empty Base Optimisation for internal_accumulator_base derivation
            CUBLASDX_DEVICE
            tmem_unified_pipeline(int                         count,
                                  GmemDescriptorA const&      in_gmem_a,
                                  CopyFunctorA                copy_a,
                                  GmemDescriptorB const&      in_gmem_b,
                                  CopyFunctorB                copy_b,
                                  byte*                       smem,
                                  PipelinedSmemLayoutA const& layout_a,
                                  PipelinedSmemLayoutB const& layout_b):
                base_type(count, in_gmem_a, copy_a, in_gmem_b, copy_b),
                internal_accumulator_base(make_internal_accumulator()) {
                static_assert(barrier_bytes % barrier_buffer_alignment_bytes == 0);
                static_assert(tmem_bytes % tmem_alignment_bytes == 0);

                auto [sliced_smem_a, sliced_smem_b, barriers, tmem_ptr] =
                    shared_memory::slice<IOTypeA, IOTypeB, barrier_storage_t, tmem_ptr_t>(
                        smem,
                        tma_buffer_alignment_bytes,
                        layout_a,
                        tma_buffer_alignment_bytes,
                        layout_b,
                        barrier_buffer_alignment_bytes,
                        barrier_bytes / barrier_buffer_alignment_bytes,
                        tmem_alignment_bytes,
                        tmem_bytes / tmem_alignment_bytes);

                // Initialize barriers for both pipeline and accumulator owned
                // members. This must be done here to allow for an expected __syncthreads()
                // in code.
                copy_smem_barriers.reset(barriers);
                compute_smem_barriers.reset(barriers + PipelineDepth);
                compute_tmem_barriers.reset(barriers + 2 * PipelineDepth);
                epilogue_tmem_barriers.reset(barriers + 2 * PipelineDepth + accumulator_stages);
                shared_tensor_storage.reset(sliced_smem_a, sliced_smem_b);

                cute::TMEM::Allocator1Sm tmem_allocator {};

                if (is_mma()) {
                    // Allocate ALL columns to be able to assume tmem_ptr == 0
                    tmem_allocator.allocate(tmem_columns, tmem_ptr);
                }

                // Only 1 thread initializes mbarriers
                if (first_compute_predicate) {
                    copy_smem_barriers.init_all();
                    compute_smem_barriers.init_all();
                    compute_tmem_barriers.init_all();
                    epilogue_tmem_barriers.init_all();
                }

                // Mandatory - change to cluster sync for 2SM execution
                __syncthreads();

                // Bind accumulator barriers
                if constexpr (accumulation == internal_accumulation) {
                    bind_accumulator(this->internal_accumulator());
                }
            }

            // Synchronize epilogue threads and deallocate TMEM. 
            // Using MMA without epilogue after it is considered an error and 
            // not handled (waiting for MMA warps)
            CUBLASDX_DEVICE
            ~tmem_unified_pipeline() {
                if (is_active()) {
                    cutlass::detail::NamedBarrierSync<num_epilogue_threads, epilogue_done_named_barrier_id>::sync();
                }
                if (is_mma()) {
                    cute::TMEM::Allocator1Sm tmem_allocator {};
                    tmem_allocator.release_allocation_lock();
                    tmem_allocator.free(0, tmem_columns);
                }
            }

            // Wait until TMA loads all data into SMEM buffers
            CUBLASDX_DEVICE
            void compute_smem_acquire() {
                PIPELINE_LOG(
                    "compute_smem_acquire tid %d stage %d phase %d\n", tid, read_cursor.index(), read_cursor.phase());
                copy_smem_barriers[read_cursor.index()].wait(read_cursor.phase());
            };

            // Signal that compute is done and the SMEM buffer can be filled with the next stage
            CUBLASDX_DEVICE
            void compute_smem_commit() {
                PIPELINE_LOG("compute_smem_commit tid %d stage %d \n", tid, read_cursor.index());

                unsigned const pipe_read = read_cursor.index();

                // NVVM code movement fence
                asm volatile("" ::: "memory");

                // In UTCMMA only a single thread commits work (the same one that delegated it)
                if (first_compute_predicate) {
                    compute_smem_barriers[pipe_read].arrive_after();
                }

                // Move stage to the next
                read_cursor.advance();
            };

            private:

            // Helper for internal asserting and preparing of the accumulator
            template<class Accumulator>
            CUBLASDX_DEVICE
            void bind_accumulator(Accumulator& accumulator) {
                static_assert(Accumulator::accumulator_stages == accumulator_stages,
                              "Accumulator and pipeline must use the same number of accumulator stages");
                static_assert(COMMONDX_STL_NAMESPACE::is_same_v<typename Accumulator::compute_tmem_barrier_t,
                                                                compute_tmem_barrier_t>,
                              "Accumulator and pipeline compute TMEM barriers must have the same type");
                static_assert(COMMONDX_STL_NAMESPACE::is_same_v<typename Accumulator::epilogue_tmem_barrier_t,
                                                                epilogue_tmem_barrier_t>,
                              "Accumulator and pipeline epilogue TMEM barriers must have the same type");
                accumulator.bind_barriers(compute_tmem_barriers, epilogue_tmem_barriers, is_mma());
            }

            // Wait until the SMEM buffer is free for the next copy
            CUBLASDX_DEVICE
            void copy_smem_acquire() {
                PIPELINE_LOG(
                    "copy_smem_acquire tid %d stage %d phase %d\n", tid, write_cursor.index(), write_cursor.phase());

                unsigned const pipe_write = write_cursor.index();
                compute_smem_barriers[pipe_write].wait(write_cursor.phase());

                // Some instructions (like TMA) require signaling barrier before 
                // launching a copy. This is in opposition to LDGSTS which requires
                // commiting afterwards
                if (should_issue_tma()) {
                    copy_smem_barriers[pipe_write].arrive_before();
                }
            };

            // Signal that:
            // 1. Copy instruction will signal current mbarrier to manifest copy being done (LDGSTS)
            // 2. All copy threads signal the mbarrier when they're done (LDG+STS)
            CUBLASDX_DEVICE
            void copy_smem_commit() {
                PIPELINE_LOG("copy_smem_commit tid %d stage %d\n", tid, write_cursor.index());

                copy_smem_barriers[write_cursor.index()].arrive_after();
                write_cursor.advance();
            };

            // Schedule a load from GlobalStage (GMEM) to SharedStage (SMEM).
            // Static/Dynamic parametrization required for preloading efficiency - we want it unrolled
            template<class GlobalStage, class SharedStage>
            CUBLASDX_DEVICE void schedule_current_ab_load(GlobalStage global_stage, SharedStage shared_stage) {
                PIPELINE_LOG("Scheduling A/B tid %d chunk %d to stage %d\n",
                             tid,
                             unsigned(global_stage),
                             unsigned(shared_stage));

                const auto copy_tid = execution_model_t::thread_copy_index(tid);

                auto write_barrier = copy_smem_barriers[shared_stage];

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

            // Batch schedule first loads, so that all possible bytes are in flight
            // This is statically unrolled - indices can be computed in compile time
            CUBLASDX_DEVICE
            void prefill() {
                PIPELINE_LOG("Copying A/B tid %d chunk %d to stage %d\n", tid, write_cursor.chunk, write_cursor.index());

                cute::for_each(cute::make_int_sequence<PipelineDepth> {}, [&](auto stage) {
                    // Wait until stage ready
                    copy_smem_acquire();
                    // Schedule
                    if (should_issue_tma()) {
                        schedule_current_ab_load(stage, write_cursor.index());
                    }
                    // Signal stage done
                    copy_smem_commit();
                });
            };

            // Copy current (dynamically indexed) stage
            CUBLASDX_DEVICE
            void copy_next() {
                PIPELINE_LOG("Copying A/B tid %d chunk %d to stage %d\n", tid, write_cursor.chunk, write_cursor.index());

                // Wait until buffer ready
                copy_smem_acquire();

                // Only a single threads issues copies in TMA
                if (should_issue_tma()) {
                    schedule_current_ab_load(write_cursor.chunk, write_cursor.index());
                }

                // Commit
                copy_smem_commit();
            };

            // Unified - all warps go through the same path, although with predicates sometimes
            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE void execute_tma_mma_loop(BlasAccumulator&& accumulator,
                                                      ALoadOp const&    a_load_op = {},
                                                      BLoadOp const&    b_load_op = {}) {

                // TMA mainloop
                if(is_tma()) {
                    // Prefill SMEM buffers
                    prefill();
                    while(this->has_copy_smem_work()) {
                        PIPELINE_LOG("tma tid: %d stage: %d\n", threadIdx.x, copy_smem_chunk());
                        // Load next
                        copy_next();
                    }
                }

                // GEMM Mainloop
                if(is_mma()) {
                    // Wait until TMEM stage is ready
                    accumulator.compute_tmem_acquire();

                    while (this->has_compute_smem_work()) {
                        PIPELINE_LOG("mma tid: %d stage: %d\n", threadIdx.x, compute_smem_chunk());
                        // Compute current
                        BLAS().execute_from_pipeline(*this, accumulator, a_load_op, b_load_op);
                    }
                }
            }

            // Execute using reusable accumulator
            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value>
                            execute_unified(BlasAccumulator&& accumulator,
                                            ALoadOp const&    a_load_op = {},
                                            BLoadOp const&    b_load_op = {}) {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");

                // This call is gated inside --> MMA and TMA paths are separate
                execute_tma_mma_loop(accumulator, a_load_op, b_load_op);
            }

            // Execute using internal accumulator - accept epilogue lambda
            template<class EpilogueFunctor, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<not is_valid_blas_accumulator<EpilogueFunctor>::value>
                            execute_unified(EpilogueFunctor const& epilogue_functor,
                                            ALoadOp const&         a_load_op = {},
                                            BLoadOp const&         b_load_op = {}) {
                static_assert(accumulation == internal_accumulation, "This method is available only for internal accumulation");

                auto& accumulator = this->internal_accumulator();

                // Gated inside --> separate TMA and MMA paths
                accumulator.clear();
                execute_tma_mma_loop(accumulator, a_load_op, b_load_op);
                
                // Signal that TMEM stage is ready to be read
                accumulator.compute_tmem_commit();

                // Perform epilogue
                if (accumulator.is_thread_active()) {
                    epilogue_functor(accumulator);
                }
            }

            CUBLASDX_DEVICE
            auto get_accumulator_impl() {
                auto acc = make_accumulator();
                bind_accumulator(acc);
                return acc;
            }

            public:

            CUBLASDX_DEVICE
            auto get_accumulator() {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");
                return get_accumulator_impl();
            }

            // Sync only threads taking part in epilogue --> can be used inside lambda/if
            CUBLASDX_DEVICE
            void epilogue_sync() {
                if (is_active()) {
                    cutlass::detail::NamedBarrierSync<num_epilogue_threads, epilogue_named_barrier_id>::sync();
                }
            }

            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator_v<BlasAccumulator>>
            execute(BlasAccumulator&& accumulator, ALoadOp const& a_load_op = {}, BLoadOp const& b_load_op = {}) {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");
                execute_unified(accumulator, a_load_op, b_load_op);
            }

            // Epilogue --> library takes care of scaffolding
            template<class BlasAccumulator, class EpilogueFunctor>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value>
            epilogue(BlasAccumulator&& accumulator,
                     EpilogueFunctor const& epilogue_functor) {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");
                // Signal TMEM stage is ready to be read
                accumulator.compute_tmem_commit();

                // If thread owns some accumulator elements, perform epilogue
                if (accumulator.is_thread_active()) {
                    epilogue_functor(accumulator);
                }
            }

            template<class EpilogueFunctor, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<not is_valid_blas_accumulator_v<EpilogueFunctor>>
            execute(EpilogueFunctor&& epilogue_functor, ALoadOp const& a_load_op = {}, BLoadOp const& b_load_op = {}) {
                static_assert(accumulation == internal_accumulation, "This method is available only for internal accumulation");
                execute_unified(epilogue_functor, a_load_op, b_load_op);
            }
        };

        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 copy_kind      CopyKind,
                 class BLAS,
                 bool IsSuggested,
                 int  AdditionalTmaAndMmaThreads,
                 class... Rest>
        struct is_pipeline<tmem_unified_pipeline<PipelineDepth,
                                                 ResultStorage,
                                                 CopyKind,
                                                 BLAS,
                                                 IsSuggested,
                                                 AdditionalTmaAndMmaThreads,
                                                 Rest...>>: cute::true_type {
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_TMEM_UNIFIED_PIPELINE_HPP
