// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_TMEM_SPECIALIZED_PIPELINE_HPP
#define CUBLASDX_DETAIL_PIPELINE_TMEM_SPECIALIZED_PIPELINE_HPP

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
#include "cublasdx/detail/pipeline/tile/tile_pipeline_traits.hpp"

namespace cublasdx {
    namespace detail {

        // Pipeline doing 3-way specialization:
        // 1. 128 threads (4 warps) responsible for epilogue
        // 2. 32 threads responsible for MMA (1 thread issuing)
        // 3. 32 threads responsible for TMA (1 thread issuing)

        // The async happens on 2 levels:
        // 1. Independent shared memory buffers --> loading and computing is independent
        // 2. Independent TMEM buffers --> computing and doing epilogue is independent

        // This structure allows for decoupling Loading/Compute and Compute/Epilogue with use of mbarriers
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
        struct tmem_specialized_pipeline :
            public tile_pipeline_base<PipelineDepth, ResultStorage, BLAS,
                                 GmemDescriptorA, CopyFunctorA, GmemDescriptorB, CopyFunctorB,
                                 PipelinedSmemLayoutA, IOTypeA, PipelinedSmemLayoutB, IOTypeB>,
            private tmem_internal_accumulator_storage<
                validate_accumulator_mode<ResultStorage>::value == internal_accumulation,
                cute::conditional_t<IsSuggested, typename BLAS::suggested_accumulator_t, typename BLAS::default_accumulator_t>> {

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
            static constexpr int      rank             = base_type::rank;

            // Config -- metainfo precomputed for readability
            using config = tmem_specialized_pipeline_config<PipelineDepth,
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

            static constexpr auto mma_instruction_kind = config::mma_instruction_kind;
            static_assert(mma_instruction_kind == instruction_type::utcmma,
                          "tmem_specialized_pipeline is only for UTCMMA instructions");

            static constexpr unsigned barrier_bytes = config::barrier_bytes;

            static constexpr int mma_warp_idx       = config::mma_warp_idx;
            static constexpr int tma_warp_idx       = config::tma_warp_idx;

            // Only 1 thread is used for TMA and UTCMMA issuing
            bool const first_copy_predicate    = cute::elect_one_sync() and (warp_idx == static_cast<unsigned>(tma_warp_idx));
            bool const first_compute_predicate = cute::elect_one_sync() and (warp_idx == static_cast<unsigned>(mma_warp_idx));

            static constexpr int additional_tma_and_mma_threads = config::additional_tma_and_mma_threads;
            static constexpr int max_threads_per_block = config::max_threads_per_block;

            // Number of threads used for epilogue described by the user as BlockDim<>
            static constexpr int num_epilogue_threads = config::num_epilogue_threads;

            // TMEM stages used for overlapping compute with epilogue
            static constexpr int accumulator_stages = config::accumulator_stages;
            // All TMEM columns reserved
            static constexpr int tmem_columns = config::tmem_columns;

            // Mainloop barriers (SMEM read/write)
            using copy_smem_barrier_t = typename config::copy_smem_barrier_t;
            using compute_smem_barrier_t = typename config::compute_smem_barrier_t;

            // Epilogue barriers (TMEM read/write)
            using epilogue_tmem_barrier_t = typename config::epilogue_tmem_barrier_t;
            using compute_tmem_barrier_t = typename config::compute_tmem_barrier_t;

            // Initialized and owned by pipeline
            barrier_group<copy_smem_barrier_t, PipelineDepth> copy_smem_barriers {};
            barrier_group<compute_smem_barrier_t, PipelineDepth> compute_smem_barriers {};

            // Allocated and initialized by pipeline, owned by accumulator
            barrier_group<compute_tmem_barrier_t, accumulator_stages> compute_tmem_barriers {};
            barrier_group<epilogue_tmem_barrier_t, accumulator_stages> epilogue_tmem_barriers {};

            // EBO
            using accumulator_t = typename config::accumulator_t;
            using internal_accumulator_base =
                tmem_internal_accumulator_storage<accumulation == internal_accumulation, accumulator_t>;

            CUBLASDX_DEVICE
            static accumulator_t make_accumulator() {
                if constexpr (IsSuggested) {
                    return BLAS().suggest_accumulator_impl();
                } else {
                    return BLAS().get_accumulator_impl();
                }
            }

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

            CUBLASDX_DEVICE
            tmem_specialized_warp_role role() const {
                return execution_model_t::role(warp_idx);
            }

            CUBLASDX_DEVICE
            bool is_epilogue() const {
                return execution_model_t::is_epilogue(warp_idx);
            }

            CUBLASDX_DEVICE
            bool is_mma() const {
                return role() == tmem_specialized_warp_role::mma;
            }

            CUBLASDX_DEVICE
            bool is_tma() const {
                return role() == tmem_specialized_warp_role::tma;
            }

            CUBLASDX_DEVICE
            bool should_issue_tma() const {
                return first_copy_predicate;
            }

            CUBLASDX_DEVICE
            bool should_issue_mma() const {
                return is_mma();
            }

            CUBLASDX_DEVICE
            tmem_specialized_pipeline(int                         count,
                                      GmemDescriptorA const&      in_gmem_a,
                                      CopyFunctorA                copy_a,
                                      GmemDescriptorB const&      in_gmem_b,
                                      CopyFunctorB                copy_b,
                                      byte*                       smem,
                                      PipelinedSmemLayoutA const& layout_a,
                                      PipelinedSmemLayoutB const& layout_b):
                base_type(count, in_gmem_a, copy_a, in_gmem_b, copy_b),
                internal_accumulator_base(make_internal_accumulator()) {
                constexpr unsigned a_alignment = tma_buffer_alignment_bytes;
                constexpr unsigned b_alignment = tma_buffer_alignment_bytes;

                static_assert(barrier_bytes % barrier_buffer_alignment_bytes == 0);
                static_assert(tmem_bytes % tmem_alignment_bytes == 0);

                auto [sliced_smem_a, sliced_smem_b, barriers, tmem_ptr] =
                    shared_memory::slice<IOTypeA, IOTypeB, barrier_storage_t, tmem_ptr_t>(
                        smem,
                        a_alignment,
                        layout_a,
                        b_alignment,
                        layout_b,
                        barrier_buffer_alignment_bytes,
                        barrier_bytes / barrier_buffer_alignment_bytes,
                        tmem_alignment_bytes,
                        tmem_bytes / tmem_alignment_bytes);

                // Set pointers
                copy_smem_barriers.reset(barriers);
                compute_smem_barriers.reset(barriers + PipelineDepth);
                compute_tmem_barriers.reset(barriers + 2 * PipelineDepth);
                epilogue_tmem_barriers.reset(barriers + 2 * PipelineDepth + accumulator_stages);
                shared_tensor_storage.reset(sliced_smem_a, sliced_smem_b);

                // Allocate all of TMEM
                cute::TMEM::Allocator1Sm tmem_allocator {};

                if (is_mma()) {
                    tmem_allocator.allocate(tmem_columns, tmem_ptr);
                }

                // Initialize barriers from pipeline
                if (should_issue_tma()) {
                    copy_smem_barriers.init_all();
                    compute_smem_barriers.init_all();
                    // Later owned by accumulator
                    compute_tmem_barriers.init_all();
                    epilogue_tmem_barriers.init_all();
                }

                // Requires to ensure mbarrier visibility
                __syncthreads();

                // If internal accumulation is used prepare the accumulator
                if constexpr (accumulation == internal_accumulation) {
                    bind_accumulator(this->internal_accumulator());
                }
            }

            CUBLASDX_DEVICE
            ~tmem_specialized_pipeline() {
                // Wait until epilogue+MMA is done
                if (is_active() || is_mma()) {
                    cutlass::detail::NamedBarrierSync<num_epilogue_threads + 32, epilogue_done_named_barrier_id>::sync();
                }
                // Deallocate TMEM from the same warp that allocated it
                if (is_mma()) {
                    cute::TMEM::Allocator1Sm tmem_allocator {};
                    tmem_allocator.release_allocation_lock();
                    tmem_allocator.free(0, tmem_columns);
                }
            }

            // Wait until the SMEM buffer for the current stage is fully loaded
            CUBLASDX_DEVICE
            void compute_smem_acquire() {
                PIPELINE_LOG(
                    "compute_smem_acquire tid %d stage %d phase %d\n", tid, read_cursor.index(), read_cursor.phase());
                copy_smem_barriers[read_cursor.index()].wait(read_cursor.phase());
            };

            // Signal that compute is done and the SMEM buffer is ready for the next copy
            CUBLASDX_DEVICE
            void compute_smem_commit() {
                PIPELINE_LOG("compute_smem_commit tid %d stage %d \n", tid, read_cursor.index());

                unsigned const pipe_read = read_cursor.index();

                // NVVM code movement fence
                asm volatile("" ::: "memory");

                if (first_compute_predicate) {
                    compute_smem_barriers[pipe_read].arrive_after();
                }

                read_cursor.advance();
            };

            private:

            // Bind internal accumulator state, done as setup once
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

            // Wait until buffer has finished computations and can be loaded into again
            CUBLASDX_DEVICE
            void copy_smem_acquire() {
                PIPELINE_LOG(
                    "copy_smem_acquire tid %d stage %d phase %d\n", tid, write_cursor.index(), write_cursor.phase());

                unsigned const pipe_write = write_cursor.index();
                compute_smem_barriers[pipe_write].wait(write_cursor.phase());
                if (should_issue_tma()) {
                    copy_smem_barriers[pipe_write].arrive_before();
                }
            };

            // Signal that copying data into the SMEM buffer is done
            CUBLASDX_DEVICE
            void copy_smem_commit() {
                PIPELINE_LOG("copy_smem_commit tid %d stage %d\n", tid, write_cursor.index());

                copy_smem_barriers[write_cursor.index()].arrive_after();
                write_cursor.advance();
            };

            // Schedule load from Global to Shared (stages passed explicitly as static or dynamic to enable efficient prefill)
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

            // Batch schedule initial load so that as much bytes as possible are in flight
            CUBLASDX_DEVICE
            void prefill() {
                assert(is_tma());
                PIPELINE_LOG("Copying A/B tid %d chunk %d to stage %d\n", tid, write_cursor.chunk, write_cursor.index());

                cute::for_each(cute::make_int_sequence<PipelineDepth> {}, [&](auto stage) {
                    copy_smem_acquire();
                    if (should_issue_tma()) {
                        schedule_current_ab_load(stage, write_cursor.index());
                    }
                    copy_smem_commit();
                });
            };

            // Schedule next copy
            CUBLASDX_DEVICE
            void copy_next() {
                assert(is_tma());
                PIPELINE_LOG("Copying A/B tid %d chunk %d to stage %d\n", tid, write_cursor.chunk, write_cursor.index());

                copy_smem_acquire();

                if (should_issue_tma()) {
                    schedule_current_ab_load(write_cursor.chunk, write_cursor.index());
                }

                copy_smem_commit();
            };

            // TMA Mainloop
            CUBLASDX_DEVICE
            void execute_tma_loop() {
                assert(is_tma());
                this->prefill();
                while (this->has_copy_smem_work()) {
                    this->copy_next();
                    PIPELINE_LOG("tma tid: %d stage: %d\n", threadIdx.x, copy_smem_chunk());
                }
            }

            // MMA Mainloop
            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE void execute_mma_loop(BlasAccumulator&& accumulator,
                                                   ALoadOp const&    a_load_op = {},
                                                   BLoadOp const&    b_load_op = {}) {
                assert(is_mma());
                while (this->has_compute_smem_work()) {
                    PIPELINE_LOG("mma tid: %d stage: %d\n", threadIdx.x, compute_smem_chunk());
                    BLAS().execute_from_pipeline(*this, accumulator, a_load_op, b_load_op);
                }
            }

            // MMA + TMA --> specialize and execute
            // Reusable accumulator --> user owns it and epilogue
            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value>
                            execute_warp_specialized(BlasAccumulator&& accumulator,
                                                     ALoadOp const&    a_load_op = {},
                                                     BLoadOp const&    b_load_op = {}) {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");

                if (is_tma()) {
                    execute_tma_loop();
                }

                if (is_mma()) {
                    accumulator.compute_tmem_acquire();
                    execute_mma_loop(accumulator, a_load_op, b_load_op);
                }
            }

            // MMA + TMA + Epilogue --> specialize and execute
            // Internal accumulator --> epilogue happens via lambda
            template<class EpilogueFunctor, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<not is_valid_blas_accumulator<EpilogueFunctor>::value>
                            execute_warp_specialized(EpilogueFunctor const& epilogue_functor,
                                                     ALoadOp const&         a_load_op = {},
                                                     BLoadOp const&         b_load_op = {}) {
                static_assert(accumulation == internal_accumulation, "This method is available only for internal accumulation");

                auto warp_role = role();
                auto& accumulator = this->internal_accumulator();

                if (is_tma()) {
                    execute_tma_loop();
                }

                if (is_mma()) {
                    accumulator.compute_tmem_acquire();
                    accumulator.clear();
                    execute_mma_loop(accumulator, a_load_op, b_load_op);
                }

                accumulator.compute_tmem_commit();

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

            // Can be used to synchronize threads inside lambda epilogue
            CUBLASDX_DEVICE
            void epilogue_sync() {
                if (is_active()) {
                    cutlass::detail::NamedBarrierSync<num_epilogue_threads, epilogue_named_barrier_id>::sync();
                }
            }

            // Reusable accumulator --> MMA + TMA without epilogue
            template<class BlasAccumulator, class ALoadOp = identity, class BLoadOp = identity>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator_v<BlasAccumulator>>
            execute(BlasAccumulator&& accumulator, ALoadOp const& a_load_op = {}, BLoadOp const& b_load_op = {}) {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");
                execute_warp_specialized(accumulator, a_load_op, b_load_op);
            }

            // Reusable accumulator epilogue --> portable and detail hiding
            template<class BlasAccumulator, class EpilogueFunctor>
            CUBLASDX_DEVICE cute::enable_if_t<is_valid_blas_accumulator<BlasAccumulator>::value>
            epilogue(BlasAccumulator&& accumulator,
                     EpilogueFunctor const& epilogue_functor) {
                static_assert(accumulation == reusable_accumulator, "This method is available only for reusable accumulators");
                accumulator.compute_tmem_commit();
                if (accumulator.is_thread_active()) {
                    epilogue_functor(accumulator);
                }
            }

            // Internal accumulation --> MMA + TMA + epilogue
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
                 int  AdditionalTmaAndMmaThreads,
                 class... Rest>
        struct is_pipeline<tmem_specialized_pipeline<PipelineDepth,
                                                    ResultStorage,
                                                    CopyKind,
                                                    BLAS,
                                                    IsSuggested,
                                                    AdditionalTmaAndMmaThreads,
                                                    Rest...>>: cute::true_type {
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_TMEM_SPECIALIZED_PIPELINE_HPP
