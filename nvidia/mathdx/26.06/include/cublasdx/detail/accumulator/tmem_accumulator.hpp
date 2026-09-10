// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_ACCUMULATOR_TMEM_ACCUMULATOR_HPP
#define CUBLASDX_DETAIL_ACCUMULATOR_TMEM_ACCUMULATOR_HPP

#include "cublasdx/detail/accumulator/accumulator_helpers.hpp"
#include "cublasdx/detail/accumulator/copy_fragment.hpp"
#include "cublasdx/detail/numerical.hpp"
#include "cublasdx/detail/pipeline/barriers/barrier_view.hpp"
#include "cublasdx/detail/pipeline/barriers/mbarrier_sync_barrier.hpp"
#include "cublasdx/detail/pipeline/barriers/mbarrier_utcmma_barrier.hpp"
#include "cublasdx/detail/pipeline/tile/tile_pipeline_stage.hpp"
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>

namespace cublasdx {
    namespace detail {
        inline constexpr int tmem_allocation_columns = 512;

        constexpr int compute_tmem_cols_per_stage(int tile_n, int sizeof_c) {
            const int columns = closest_next_power_of_two(tile_n * sizeof_c / 4);
            return columns < 32 ? 32 : columns;
        }

        // TMEM has 128 lanes; when the accumulator M dimension spans multiple
        // MMA atoms (M > atom_m), each additional M block occupies its own set
        // of columns, so a stage's footprint scales with the M block count.
        constexpr int compute_tmem_stage_columns(int m_blocks, int tile_n, int sizeof_c) {
            return m_blocks * compute_tmem_cols_per_stage(tile_n, sizeof_c);
        }

        constexpr int compute_max_num_tmem_stages(int m_blocks, int tile_n, int sizeof_c) {
            return tmem_allocation_columns / compute_tmem_stage_columns(m_blocks, tile_n, sizeof_c);
        }

        constexpr int compute_tmem_chunk_tile_n(int tile_n, int atom_tile_n) {
            constexpr int preferred_tmem_chunk_tile_n = 32;
            constexpr int tmem_chunk_tile_n_granularity = 8;
            const int max_chunk = atom_tile_n < preferred_tmem_chunk_tile_n ? atom_tile_n : preferred_tmem_chunk_tile_n;
            const int aligned_max_chunk =
                (max_chunk / tmem_chunk_tile_n_granularity) * tmem_chunk_tile_n_granularity;

            // Keep chunks close to 32 columns, but require them to tile both
            // the accumulator N dimension and the underlying UMMA atom.
            for (int chunk = aligned_max_chunk; chunk >= tmem_chunk_tile_n_granularity;
                 chunk -= tmem_chunk_tile_n_granularity) {
                if ((tile_n % chunk == 0) && (atom_tile_n % chunk == 0)) {
                    return chunk;
                }
            }
            return 0;
        }


        // Infer TMEM fragment type from MMA, then infer copy type based on:
        // - MMA 
        // - Problem Shape
        template<typename InputTypeC, class SlicedMMA, class ShapeMN, class AtomShapeMNK, int NumStages>
        struct tmem_fragments_and_copy {
            // Construct faux CTA-wide tensor of problem shape (Size<M, N>)
            using cta_c_tensor_t =
                decltype(cute::make_tensor(cute::make_gmem_ptr<InputTypeC>(static_cast<InputTypeC*>(nullptr)),
                                           cute::make_layout(ShapeMN {})));
            // Use UTCMMA partitioning to group appropriate elements together
            using partitioned_cta_c_tensor_t = decltype(SlicedMMA().partition_C(cta_c_tensor_t {}));
            // Based on the above type construct TMEM fragment
            using mma_tmem_fragment_t =
                decltype(SlicedMMA().make_fragment_C(cute::declval<partitioned_cta_c_tensor_t>()));

            using copy_atom_base_1x_small_t = cute::conditional_t<sizeof(InputTypeC) == sizeof(float),
                                                        cute::SM100_TMEM_LOAD_16dp32b1x,
                                                        cute::SM100_TMEM_LOAD_16dp32b1x_16b>;
            using copy_atom_base_1x_big_t   = cute::conditional_t<sizeof(InputTypeC) == sizeof(float),
                                                        cute::SM100_TMEM_LOAD_32dp32b1x,
                                                        cute::SM100_TMEM_LOAD_32dp32b1x_16b>;
            
            // Depending on whether UTCMMA has M=64 or M=128 different type of base atom must be chosen
            // for correctness
            using tmem_chunk_copy_atom_t = cute::conditional_t<cute::get<0>(AtomShapeMNK {}) == 128,
                                                               copy_atom_base_1x_big_t,
                                                               copy_atom_base_1x_small_t>;

            // Construct appropriate copy
            using tmem_copy_t =
                decltype(cute::make_tmem_copy(tmem_chunk_copy_atom_t {}, cute::declval<mma_tmem_fragment_t>()));
            using sliced_tmem_copy_t = decltype(cute::declval<tmem_copy_t>().get_slice(0));

            // Based on UTCMMA copy and output tensor partition, construct appropriate RMEM fragment type
            static constexpr int m_atom_blocks =
                cute::get<0>(ShapeMN {}) / cute::get<0>(AtomShapeMNK {});
            static constexpr int columns_per_stage =
                compute_tmem_stage_columns(m_atom_blocks, cute::get<1>(ShapeMN {}), sizeof(InputTypeC));
            using tmem_stage_layout_t =
                cute::Layout<cute::Shape<cute::Int<NumStages>>, cute::Stride<cute::Int<columns_per_stage>>>;
            using tmem_fragment_t =
                decltype(cute::make_tensor(cute::declval<mma_tmem_fragment_t>().data(),
                                           cute::prepend(cute::declval<mma_tmem_fragment_t>().layout(),
                                                         tmem_stage_layout_t {})));
            using register_fragment_t = decltype(cute::make_fragment_like(
                sliced_tmem_copy_t(0).partition_D(cute::declval<partitioned_cta_c_tensor_t>())));
        };

        template<class TiledMMA,
                 instruction_type InstructionKind,
                 class LoadInstruction,
                 class StoreInstruction,
                 class ShapeMN,
                 class InputTypeC,
                 class Alignment,
                 class HasStaticBlockDim,
                 class BlockSize,
                 class AccumulatorSM = cute::Int<1000>>
        struct tmem_accumulator {
            using atom_shape_mnk = typename TiledMMA::AtomShape_MNK;
            using value_type = InputTypeC;
            static constexpr int tmem_columns = tmem_allocation_columns;
            static constexpr int m_atom_blocks =
                cute::get<0>(ShapeMN {}) / cute::get<0>(atom_shape_mnk {});
            static_assert(m_atom_blocks >= 1, "Accumulator M dimension is smaller than the MMA atom M");
            static constexpr int accumulator_stages =
                compute_max_num_tmem_stages(m_atom_blocks, cute::get<1>(ShapeMN {}), sizeof(InputTypeC));
            static_assert(accumulator_stages >= 1,
                          "Accumulator tile (M blocks x N columns) exceeds the TMEM capacity of one SM");
            using epilogue_tmem_barrier_t = mbarrier_utcmma_barrier<AccumulatorSM::value, 1>;
            using compute_tmem_barrier_t = mbarrier_sync_barrier<AccumulatorSM::value, BlockSize::value>;

            static constexpr Alignment         alignment            = {};
            static constexpr HasStaticBlockDim has_static_block_dim = {};
            using sliced_mma_t   = decltype(TiledMMA().get_thread_slice(cute::declval<unsigned>()));
            using coord_tensor_t = decltype(cute::make_identity_tensor(ShapeMN {}));

            static constexpr bool is_partition_divisible =
                decltype(cute::evenly_divides(ShapeMN {},
                                              cute::select<0, 1>(cute::tile_shape(TiledMMA {}))))::value;
            static_assert(is_partition_divisible, "TMEM accumulators do not support predicated tiles");

            // Number of MMA atom blocks that tile the accumulator M/N dimensions
            static constexpr int atom_block_count_inside_tile_m =
                cute::get<0>(ShapeMN {}) / cute::get<0>(atom_shape_mnk {});
            static constexpr int atom_block_count_inside_tile_n =
                cute::get<1>(ShapeMN {}) / cute::get<1>(atom_shape_mnk {});

            // Retrieve appropriate types of TMEM/RMEM fragments and copy atom between them
            using fragments_and_copy =
                tmem_fragments_and_copy<InputTypeC, sliced_mma_t, ShapeMN, atom_shape_mnk, accumulator_stages>;
            using tmem_fragment_t = typename fragments_and_copy::tmem_fragment_t;
            using tmem_copy_t = typename fragments_and_copy::tmem_copy_t;
            using sliced_tmem_copy_t = typename fragments_and_copy::sliced_tmem_copy_t;
            using register_fragment_t = typename fragments_and_copy::register_fragment_t;
            using tmem_chunk_copy_atom_t = typename fragments_and_copy::tmem_chunk_copy_atom_t;

            // Decide on TMEM chunking meta-parameters. Chunk N must divide
            // the accumulator N dimension; TMEM copy granularity is 8 columns.
            static constexpr int atom_tile_n = static_cast<int>(cute::get<1>(atom_shape_mnk {}));
            static constexpr int accumulator_tile_n = static_cast<int>(cute::get<1>(ShapeMN {}));
            static_assert(accumulator_tile_n % 8 == 0, "TMEM accumulator N dimension must be a multiple of 8");
            static_assert(atom_tile_n % 8 == 0, "TMEM atom N dimension must be a multiple of 8");
            static constexpr int tmem_chunk_tile_n = compute_tmem_chunk_tile_n(accumulator_tile_n, atom_tile_n);
            static_assert(tmem_chunk_tile_n >= 8 && tmem_chunk_tile_n % 8 == 0,
                          "TMEM chunk width must be a positive multiple of 8");
            static_assert(accumulator_tile_n % tmem_chunk_tile_n == 0,
                          "TMEM chunk width must divide the accumulator N dimension");
            static_assert(atom_tile_n % tmem_chunk_tile_n == 0, "TMEM chunk width must divide the atom N dimension");
            static_assert((accumulator_tile_n * sizeof(InputTypeC)) % 4 == 0,
                          "TMEM accumulator requires tile_n * sizeof(C) divisible by 4");

            // Size of TMEM tile to copy at once --> same as atom size in M (not tile size in M)
            // Chunking for copying TMEM->RMEM (piece by piece)
            static constexpr int tmem_chunk_tile_m = static_cast<int>(cute::get<0>(atom_shape_mnk {}));
            using tmem_chunk_m_dim_t = cute::Int<tmem_chunk_tile_m>;
            using tmem_chunk_tile_t = cute::Shape<tmem_chunk_m_dim_t, cute::Int<tmem_chunk_tile_n>>;

            TiledMMA tiled_mma = {};

            // Dynamic values
            unsigned     thr_idx;
            sliced_mma_t sliced_mma;

            // Has place for TMEM pointer
            tmem_fragment_t accumulator_storage;

            // TMEM pipeline stage
            mutable cutlass::PipelineState<accumulator_stages> acc_pipe_state {};
            pipeline_stage_cursor<accumulator_stages> mma_stage {};

            // WARNING: this is assumed to be 0 because we are allocating 
            // ALL of TMEM. Otherwise this assumption cannot be made.
            static constexpr uint32_t tmem_base_index = 0;

            // TMEM staging barriers:
            // - compute_tmem: wait until epilogue has released a TMEM chunk
            // - epilogue_tmem: wait until compute has committed a TMEM chunk
            barrier_group<compute_tmem_barrier_t, accumulator_stages> compute_tmem_barriers {};
            barrier_group<epilogue_tmem_barrier_t, accumulator_stages> epilogue_tmem_barriers {};
            // Only thread responsible for launching UTCMMA can issue final commit 
            // this flag is responsible for this predicate
            bool should_compute_tmem_commit_ = false;

            // C += A * B on next MMA
            CUBLASDX_DEVICE
            void turn_on_accumulation() {
                static_assert(InstructionKind == instruction_type::gmma ||
                                  InstructionKind == instruction_type::utcmma,
                              "This functionality is only available for UMMA");
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            }

            // C = A * B on next MMA
            CUBLASDX_DEVICE
            void turn_off_accumulation() {
                static_assert(InstructionKind == instruction_type::gmma ||
                                  InstructionKind == instruction_type::utcmma,
                              "This functionality is only available for UMMA");
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
            }

            // Construct from thread_idx
            CUBLASDX_DEVICE
            tmem_accumulator(unsigned const thread_idx):
                thr_idx(thread_idx),
                sliced_mma(TiledMMA().get_slice(0u)) {
                mma_stage = pipeline_stage_cursor<accumulator_stages>::producer_start();
                accumulator_storage.data() = tmem_base_index;
                clear();
            }

            // The only cooperation point between pipeline and accumulator
            // this needs to exist because TMEM / barriers init should happen at the top 
            // of the kernel and requires a syncthreads (to propagate mbarrier state in smem)
            CUBLASDX_DEVICE
            void bind_barriers(barrier_group<compute_tmem_barrier_t, accumulator_stages> compute_barriers,
                               barrier_group<epilogue_tmem_barrier_t, accumulator_stages> epilogue_barriers,
                               bool should_compute_tmem_commit) {
                compute_tmem_barriers        = compute_barriers;
                epilogue_tmem_barriers       = epilogue_barriers;
                should_compute_tmem_commit_ = should_compute_tmem_commit;
            }

            // Wait until epilogue on current stage is finished
            CUBLASDX_DEVICE
            void compute_tmem_acquire() {
                assert(compute_tmem_barriers.data() != nullptr);
                compute_tmem_barriers[mma_stage.index()].wait(mma_stage.phase());
            }

            // Get TMEM chunk to accumulate into (must be acquired before)
            CUBLASDX_DEVICE
            auto compute_accumulator() const {
                auto const coord = cute::prepend(
                    cute::repeat<cute::rank_v<tmem_fragment_t> - 1>(cute::_),
                    mma_stage.index());
                return accumulator_storage(coord);
            }

            // Get TMEM chunk to perform epilogue on (must be acquired first)
            CUBLASDX_DEVICE
            auto epilogue_accumulator() const {
                auto const coord = cute::prepend(
                    cute::repeat<cute::rank_v<tmem_fragment_t> - 1>(cute::_),
                    acc_pipe_state.index());
                return accumulator_storage(coord);
            }

            // This is required because in reusable accumulation mode 
            // library can't know when user wants to finalize accumulation
            // into a single accumulator. 
            // Why can't this be hidden inside result extractors?
            // - to enable Numba CUDA a non-lambda access method to 
            // epilogue must be allowed. With cuBLASDx 0.5.0 accumulator.is_thread_active()
            // was decided to be such entry point. If result getter is called inside 
            // is_thread_active() then MMA warp has no access to it and can't issue
            // tcgen05.commit
            CUBLASDX_DEVICE
            void compute_tmem_commit() {
                if (!should_compute_tmem_commit_) {
                    return;
                }
                assert(epilogue_tmem_barriers.data() != nullptr);
                const auto stage      = mma_stage.index();
                const auto full_phase = mma_stage.phase() ^ 1;
                if (cute::elect_one_sync()) {
                    epilogue_tmem_barriers[stage].arrive_after();
                }
                epilogue_tmem_barriers[stage].wait(full_phase);
                mma_stage.advance();
            }

            CUBLASDX_DEVICE
            void finish_accumulation() {
                compute_tmem_commit();
            }

            // Wait until epilogue warpgroup finalized epilogue on 
            // current TMEM chunk
            CUBLASDX_DEVICE
            void epilogue_tmem_acquire() const {
                assert(is_thread_active());
                assert(epilogue_tmem_barriers.data() != nullptr);
                epilogue_tmem_barriers[acc_pipe_state.index()].wait(acc_pipe_state.phase());
            }

            // Signal that epilogue has been finished on current epilogue chunk
            CUBLASDX_DEVICE
            void epilogue_tmem_commit() const {
                assert(is_thread_active());
                assert(compute_tmem_barriers.data() != nullptr);
                cutlass::arch::fence_view_async_tmem_load();
                compute_tmem_barriers[acc_pipe_state.index()].arrive_after();
                ++acc_pipe_state;
            }

            // GCC7 protection
            CUBLASDX_DEVICE
            constexpr auto is_predicated() const {
                return cute::conditional_return<is_partition_divisible>(cute::false_type {}, cute::true_type {});
            }

            CUBLASDX_DEVICE
            constexpr auto get_alignment() const { return alignment; }

            // Effectively is_epilogue()
            CUBLASDX_DEVICE
            auto is_thread_active() const {
                static_assert(BlockSize::value == 128, "UTCMMA can be used only with 128 threads");
                return thr_idx < BlockSize::value;
            }

            // Internal --> expose pre-computed copy atom to fragment copying functions
            CUBLASDX_DEVICE
            auto make_tiled_load_copy_c() const {
                auto tiled_copy = cute::make_tiled_copy_C(cute::Copy_Atom<LoadInstruction, InputTypeC> {}, sliced_mma);
                auto thr_copy   = tiled_copy.get_thread_slice(thr_idx);
                return cute::make_tuple(tiled_copy, thr_copy);
            }

            // Internal --> expose pre-computed store atom to fragment copying functions
            CUBLASDX_DEVICE
            auto make_tiled_store_copy_c() const {
                auto tiled_copy = cute::make_tiled_copy_C(cute::Copy_Atom<StoreInstruction, InputTypeC> {}, sliced_mma);
                auto thr_copy   = tiled_copy.get_thread_slice(thr_idx);
                return cute::make_tuple(tiled_copy, thr_copy);
            }

            // helper
            CUBLASDX_DEVICE
            auto make_sliced_tmem_copy() const {
                return tmem_copy_t{}.get_slice(thr_idx);
            }

            // TMEM is like SMEM output for tensor cores
            // if all of it is copied at once to RMEM then a register spill
            // may happen destroying performance. It's best to stage copying
            // and do it chunk by chunk so that:
            // - vectorization is still allowed
            // - no register spills happen
            //
            // Index-space glossary for the chunked readout (used here and in
            // process_and_store / reduce_and_store below):
            //   atom_block_inside_tile_index_{m,n}  - which MMA atom block of the accumulator tile,
            //                                         in [0, atom_block_count_inside_tile_{m,n})
            //   chunk_inside_atom_block_index_{m,n} - which chunk INSIDE that atom block
            //   chunk_inside_tile_index_{m,n} - chunk coordinates in the whole tile:
            //       chunk_inside_tile_index_m = atom_block_inside_tile_index_m * (M chunks per atom)
            //                                   + chunk_inside_atom_block_index_m  (same for n)
            //   element_inside_chunk_index    - element within a chunk, per thread
            // Chunk shape is tmem_chunk_tile_t = (atom_m, tmem_chunk_tile_n), so the
            // M-chunk count per atom block is atom_m / atom_m == 1, so
            // chunk_inside_tile_index_m effectively enumerates M atom blocks.
            template<class AccM, class AccN>
            CUBLASDX_DEVICE auto make_tmem_chunk_source(AccM atom_block_inside_tile_index_m,
                                                        AccN atom_block_inside_tile_index_n) const {
                // Current-stage TMEM fragment has modes ((v_m, v_n), MMA_M, MMA_N):
                // mode 0 spans one atom's (rows x columns), modes 1/2 select the
                // atom block. Selecting (atom_block_inside_tile_index_m, atom_block_inside_tile_index_n)
                // leaves atom_block_accumulator as one atom block shaped (atom_m, atom_n)
                // in TMEM-address space.
                auto atom_block_accumulator = epilogue_accumulator()(
                    cute::make_coord(cute::_, cute::_), atom_block_inside_tile_index_m, atom_block_inside_tile_index_n);
                // flat_divide by the chunk tile: atom_block_chunks has modes
                // (chunk_m, chunk_n, rest_m, rest_n) where rest_m/rest_n count the
                // chunks inside this atom block (rest_m == 1, rest_n == atom_n / chunk_n).
                auto atom_block_chunks = cute::flat_divide(atom_block_accumulator, tmem_chunk_tile_t {});
                // Tiled TMEM->RMEM copy built over a single chunk (rest coords (0, 0))
                auto tiled_tmem_to_register_copy = cute::make_tmem_copy(
                    tmem_chunk_copy_atom_t {},
                    atom_block_chunks(cute::_, cute::_, cute::Int<0> {}, cute::Int<0> {}));
                // Slice copy
                auto thread_tmem_to_register_copy = tiled_tmem_to_register_copy.get_slice(thr_idx);
                // Per-thread source partition: thread_tmem_chunks has modes
                // ((copy atom v), copy_m, copy_n, rest_m, rest_n); modes 3 and 4 are
                // the chunk grid of this atom block, hence the size<3>/size<4> loop
                // bounds in the readout loops below.
                auto thread_tmem_chunks = thread_tmem_to_register_copy.partition_S(atom_block_chunks);
                auto dummy_d = cute::make_tensor(cute::make_gmem_ptr<InputTypeC>(nullptr),
                                                 cute::make_layout(tmem_chunk_tile_t {}));
                // Per-thread element count/shape of ONE chunk (e.g. 32 elements:
                // each thread owns one TMEM lane of the chunk, one element per column)
                auto tmem_chunk_shape = cute::shape(thread_tmem_to_register_copy.partition_D(dummy_d));
                // Return all at once
                return cute::make_tuple(
                    tiled_tmem_to_register_copy, thread_tmem_to_register_copy, thread_tmem_chunks, tmem_chunk_shape);
            }

            // Helper to get required chunk with one-liner
            template<class TmemChunkSource, class Coord>
            CUBLASDX_DEVICE
            auto get_tmem_chunk(TmemChunkSource const& source, Coord const& coord) const {
                // Input here is output from make_tmem_chunk_source:
                // 0 -> tiled_tmem_to_register_copy
                // 1 -> thread_tmem_to_register_copy
                // 2 -> thread_tmem_chunks (TMEM partitioned by sliced copy)
                //      ((copy atom v), copy_m, copy_n, rest_m, rest_n);
                // 3 -> tmem chunk shape
                auto rmem_output_for_chunk = cute::make_tensor<InputTypeC>(cute::get<3>(source)); // chunk shape
                cute::copy(cute::get<0>(source), // tiled copy
                           // ((copy atom v), copy_m, copy_n, rest_m, rest_n);
                           cute::get<2>(source)(cute::_, cute::_, cute::_,
                                                cute::get<0>(coord),
                                                cute::get<1>(coord)),
                           rmem_output_for_chunk);
                return rmem_output_for_chunk;
            }

            // Partitions an (M, N)-shaped tensor exactly like the per-thread
            // register fragment. This defines THE fragment index space: iterating
            // the result linearly walks, for each thread, all N columns of M atom
            // block 0 first, then all N columns of block 1, and so on
            // (fragment slot j = atom_block_inside_tile_index_m * N + column for this
            // thread's lanes).
            // register_fragment_t is make_fragment_like() of this partition, and
            // copy_fragment pairs fragment elements with tensor elements through it,
            // so every producer of fragment indices (see fragment_chunks in the
            // direct-output reduce_and_store branch) must enumerate in this order.
            template<class CTensor>
            CUBLASDX_DEVICE auto partition_like_C(CTensor&& ctensor) const {
                return make_sliced_tmem_copy().partition_D(sliced_mma.partition_C(static_cast<CTensor&&>(ctensor)));
            }

            // Fragment linear index -> (m, n) coordinate in the accumulator tile,
            // per the partition_like_C order described above.
            template<class... Coords>
            CUBLASDX_DEVICE auto map_fragment_index(Coords&&... coords) const {
                auto thr_coord = partition_like_C(coord_tensor_t {});
                return thr_coord(static_cast<Coords&&>(coords)...);
            }

            // Checks if an (m, n) coordinate lies inside the (M, N) problem shape.
            template<class Coord>
            CUBLASDX_DEVICE bool is_coord_in_bounds(Coord const& coord) const {
                return cute::elem_less(coord, ShapeMN {});
            }

            template<class... Coords>
            CUBLASDX_DEVICE bool is_index_in_bounds(Coords&&... coords) const {
                return is_coord_in_bounds(map_fragment_index(static_cast<Coords&&>(coords)...));
            }

            CUBLASDX_DEVICE
            auto make_empty_fragment() const {
                auto ret = cute::make_fragment_like(register_fragment_t {});
                cute::clear(ret);
                return ret;
            }

            template<class FromEngine, class FromLayout, class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_copy(tensor<FromEngine, FromLayout> const& tS,
                                                    tensor<ToEngine, ToLayout>&           tD) const {
                assert(is_thread_active());
                ::cublasdx::detail::copy_fragment<Alignment::value>(tS, tD, *this);
            }

            template<class FromEngine, class FromLayout, class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_copy(tensor<FromEngine, FromLayout> const& tS,
                                                    tensor<ToEngine, ToLayout>&&          tD) const {
                partition_and_copy(tS, tD);
            }

            template<class FromEngine,
                     class FromLayout,
                     cute::enable_if_t<cute::is_smem_v<FromEngine> or cute::is_gmem_v<FromEngine>>* = nullptr>
            CUBLASDX_DEVICE auto make_partition_and_copy(tensor<FromEngine, FromLayout> const& tS) const {
                assert(is_thread_active());
                auto frg = cute::make_fragment_like(partition_like_C(tS));
                ::cublasdx::detail::copy_fragment<Alignment::value>(tS, frg, *this);
                return frg;
            }

            // Get results as new fragment
            CUBLASDX_DEVICE
            auto get_results() const {
                assert(is_thread_active());
                register_fragment_t register_fragment;
                get_results(register_fragment);
                return register_fragment;
            }

            // Get results copied into provided fragment
            template<class Engine, class Layout>
            CUBLASDX_DEVICE
            void get_results(cublasdx::tensor<Engine, Layout>& out) const {
                assert(is_thread_active());
                epilogue_tmem_acquire();
                auto atom_block_accumulator = epilogue_accumulator();
                Tensor tDtAcc = make_sliced_tmem_copy().partition_S(atom_block_accumulator);
                cute::copy(tmem_copy_t {}, tDtAcc, out);
                epilogue_tmem_commit();
            }

            // Accept mutable temporaries
            template<class Engine, class Layout>
            CUBLASDX_DEVICE
            void get_results(cublasdx::tensor<Engine, Layout>&& out) const {
                get_results(out);
            }

            CUBLASDX_DEVICE
            void clear() { tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero; }

            CUBLASDX_DEVICE
            constexpr auto size() const { return cute::size(register_fragment_t {}); }

            template<class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_store(tensor<ToEngine, ToLayout>& tD) const {
                process_and_store(tD, identity {});
            }

            template<class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_store(tensor<ToEngine, ToLayout>&& tD) const {
                partition_and_store(tD);
            }


            // Apply user_data(i) = lambda(output(i))
            // preferably chunk by chunk to avoid RMEM spilling into GMEM
            template<class ToEngine, class ToLayout, class ProcessOp>
            CUBLASDX_DEVICE void process_and_store(tensor<ToEngine, ToLayout>& tD,
                                                   ProcessOp&&                 process_op) const {
                assert(is_thread_active());
                using d_engine_t = typename ToEngine::value_type;

                static constexpr bool is_mem_output =
                    cute::is_smem_v<ToEngine> || cute::is_gmem_v<ToEngine>;
                if constexpr (is_mem_output) {
                    epilogue_tmem_acquire();

                    // The FULL (M, N) output tile divided into chunk tiles: modes
                    // (chunk_m, chunk_n, REST_M, REST_N), where REST_M / REST_N count
                    // chunks across the whole accumulator (all atom blocks), i.e.
                    // REST_M == atom_block_count_inside_tile_m * size<3>(thread_tmem_chunks) and
                    // REST_N == atom_block_count_inside_tile_n * size<4>(thread_tmem_chunks).
                    auto output_chunks = cute::flat_divide(tD, tmem_chunk_tile_t {});
                    constexpr int max_vec_bits = cute::gcd(
                        Alignment::value * 8, cute::max_alignment(ToLayout {}) * cute::sizeof_bits_v<d_engine_t>);

                    auto const store_chunk = [&](auto const& output_chunk_slice, auto const& tmem_chunk) {
                        auto d_chunk = cute::make_tensor<d_engine_t>(cute::shape(tmem_chunk));

                        CUTE_UNROLL
                        for (int element_inside_chunk_index = 0; element_inside_chunk_index < cute::size(tmem_chunk);
                             ++element_inside_chunk_index) {
                            d_chunk(element_inside_chunk_index) =
                                static_cast<d_engine_t>(process_op(tmem_chunk(element_inside_chunk_index)));
                        }

                        cute::copy(
                            cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits> {},
                            cute::make_tensor(d_chunk.data(), cute::flatten(d_chunk.layout())),
                            cute::make_tensor(output_chunk_slice.data(), cute::flatten(output_chunk_slice.layout())));
                    };

                    // Iterate over actual chunk size
                    CUTE_NO_UNROLL
                    for (int atom_block_inside_tile_index_n = 0;
                         atom_block_inside_tile_index_n < atom_block_count_inside_tile_n;
                         ++atom_block_inside_tile_index_n) {
                        CUTE_UNROLL
                        for (int atom_block_inside_tile_index_m = 0;
                             atom_block_inside_tile_index_m < atom_block_count_inside_tile_m;
                             ++atom_block_inside_tile_index_m) {
                            // Input here is output from make_tmem_chunk_source:
                            // 0 -> tiled_tmem_to_register_copy
                            // 1 -> thread_tmem_to_register_copy
                            // 2 -> thread_tmem_chunks (TMEM partitioned by sliced copy)
                            //      ((copy atom v), copy_m, copy_n, rest_m, rest_n);
                            // 3 -> tmem chunk shape
                            auto tmem_source =
                                make_tmem_chunk_source(atom_block_inside_tile_index_m, atom_block_inside_tile_index_n);
                            auto thread_tmem_to_register_copy = cute::get<1>(tmem_source);
                            // Per-thread TMEM source of this atom block:
                            // ((copy v), copy_m, copy_n, rest_m, rest_n) - modes 3/4 are
                            // the chunk grid WITHIN the block (see make_tmem_chunk_source).
                            auto thread_tmem_chunks = cute::get<2>(tmem_source);
                            // Per-thread destination over the WHOLE output tile:
                            // ((copy v), copy_m, copy_n, REST_M, REST_N) - modes 3/4 are the
                            // global chunk grid, addressed below with chunk_inside_tile_index_{m,n}.
                            auto thread_output_chunks = thread_tmem_to_register_copy.partition_D(output_chunks);

                            // Iterate over chunk tiling
                            CUTE_NO_UNROLL
                            for (int chunk_inside_atom_block_index_n = 0;
                                 chunk_inside_atom_block_index_n < cute::size<4>(thread_tmem_chunks);
                                 ++chunk_inside_atom_block_index_n) {
                                CUTE_UNROLL
                                for (int chunk_inside_atom_block_index_m = 0;
                                     chunk_inside_atom_block_index_m < cute::size<3>(thread_tmem_chunks);
                                     ++chunk_inside_atom_block_index_m) {
                                    // Chunk coordinates in the whole accumulator; TMEM is read
                                    // with the block-local chunk index, memory is written with
                                    // the tile-global one - the destination slice is
                                    // coordinate-addressed, so no linear index is needed here.
                                    const int chunk_inside_tile_index_m =
                                        atom_block_inside_tile_index_m * cute::size<3>(thread_tmem_chunks) +
                                        chunk_inside_atom_block_index_m;
                                    const int chunk_inside_tile_index_n =
                                        atom_block_inside_tile_index_n * cute::size<4>(thread_tmem_chunks) +
                                        chunk_inside_atom_block_index_n;
                                    auto output_chunk_slice = thread_output_chunks(cute::_,
                                                                                   cute::_,
                                                                                   cute::_,
                                                                                   chunk_inside_tile_index_m,
                                                                                   chunk_inside_tile_index_n);
                                    auto tmem_chunk         = get_tmem_chunk(tmem_source,
                                                                     cute::make_coord(chunk_inside_atom_block_index_m,
                                                                                      chunk_inside_atom_block_index_n));
                                    store_chunk(output_chunk_slice, tmem_chunk);
                                }
                            }
                        }
                    }

                    epilogue_tmem_commit();
                } else {
                    // TODO: Consider RMEM backed reduce_and_store
                    static_assert(is_mem_output, "UTCMMA process_and_store output must be memory-backed");
                }
            }

            // Accept mutable temporaries
            template<class ToEngine, class ToLayout, class ProcessOp>
            CUBLASDX_DEVICE void process_and_store(tensor<ToEngine, ToLayout>&& tD,
                                                   ProcessOp&&                  process_op) const {
                process_and_store(tD, static_cast<ProcessOp&&>(process_op));
            }

            // Apply output(m, n) = lambda(result(m, n), output(m, n), (m, n))
            // preferably chunk by chunk to avoid RMEM spilling into GMEM.
            // The lambda's third argument is the element's (m, n) coordinate in
            // the accumulator tile (what map_fragment_index used to be applied to).
            template<class OutEngine, class OutLayout, class Lambda>
            CUBLASDX_DEVICE void reduce_and_store(tensor<OutEngine, OutLayout>& output,
                                                  Lambda&&                      lambda) const {
                assert(is_thread_active());
                using d_engine_t = typename OutEngine::value_type;

                // Support both SMEM/GMEM/RMEM inouts
                static constexpr bool is_mem_output =
                    cute::is_smem_v<OutEngine> || cute::is_gmem_v<OutEngine>;
                static constexpr bool is_direct_output =
                    cute::is_static_v<decltype(cute::size(OutLayout {}))> &&
                    cute::size(OutLayout {}) == cute::size(register_fragment_t {});
                static constexpr bool is_supported_output = is_mem_output || is_direct_output;
                if constexpr (is_supported_output) {
                    epilogue_tmem_acquire();

                    // Iterate over actual chunk size
                    CUTE_NO_UNROLL
                    for (int atom_block_inside_tile_index_n = 0;
                         atom_block_inside_tile_index_n < atom_block_count_inside_tile_n;
                         ++atom_block_inside_tile_index_n) {
                        CUTE_UNROLL
                        for (int atom_block_inside_tile_index_m = 0;
                             atom_block_inside_tile_index_m < atom_block_count_inside_tile_m;
                             ++atom_block_inside_tile_index_m) {
                            auto tmem_source =
                                make_tmem_chunk_source(atom_block_inside_tile_index_m, atom_block_inside_tile_index_n);
                            auto thread_tmem_to_register_copy = cute::get<1>(tmem_source);
                            auto thread_tmem_chunks = cute::get<2>(tmem_source);
                            if constexpr (is_mem_output) {
                                // Same structure as in process_and_store: output_chunks divides the
                                // WHOLE (M, N) output into chunks, and thread_output_chunks is the per-thread
                                // view ((copy v), copy_m, copy_n, REST_M, REST_N) whose modes
                                // 3/4 form the global chunk grid.
                                auto output_chunks = cute::flat_divide(output, tmem_chunk_tile_t {});
                                auto thread_output_chunks = thread_tmem_to_register_copy.partition_D(output_chunks);
                                // The (m, n) coordinate of every output element, obtained by
                                // partitioning an identity tensor of the tile shape EXACTLY like
                                // the output; coord_chunk(e) below is the tile-local coordinate
                                // handed to the user lambda.
                                auto coord_chunks = cute::flat_divide(coord_tensor_t {}, tmem_chunk_tile_t {});
                                auto thread_coord_chunks = thread_tmem_to_register_copy.partition_D(coord_chunks);
                                auto c_chunk = cute::make_tensor<d_engine_t>(cute::get<3>(tmem_source));
                                constexpr int max_vec_bits =
                                    cute::gcd(Alignment::value * 8,
                                              cute::max_alignment(OutLayout {}) * cute::sizeof_bits_v<d_engine_t>);

                                // Iterate over chunk tiling
                                CUTE_NO_UNROLL
                                for (int chunk_inside_atom_block_index_n = 0;
                                     chunk_inside_atom_block_index_n < cute::size<4>(thread_tmem_chunks);
                                     ++chunk_inside_atom_block_index_n) {
                                    CUTE_UNROLL
                                    for (int chunk_inside_atom_block_index_m = 0;
                                         chunk_inside_atom_block_index_m < cute::size<3>(thread_tmem_chunks);
                                         ++chunk_inside_atom_block_index_m) {
                                        const int chunk_inside_tile_index_m =
                                            atom_block_inside_tile_index_m * cute::size<3>(thread_tmem_chunks) +
                                            chunk_inside_atom_block_index_m;
                                        const int chunk_inside_tile_index_n =
                                            atom_block_inside_tile_index_n * cute::size<4>(thread_tmem_chunks) +
                                            chunk_inside_atom_block_index_n;
                                        // Both slices address the chunk's true (M, N) location;
                                        // coord_chunk carries the per-element (m, n) coordinates
                                        // for the user lambda.
                                        auto output_chunk_slice = thread_output_chunks(cute::_,
                                                                                       cute::_,
                                                                                       cute::_,
                                                                                       chunk_inside_tile_index_m,
                                                                                       chunk_inside_tile_index_n);
                                        auto coord_chunk        = thread_coord_chunks(cute::_,
                                                                                cute::_,
                                                                                cute::_,
                                                                                chunk_inside_tile_index_m,
                                                                                chunk_inside_tile_index_n);

                                        cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits> {},
                                                   cute::make_tensor(output_chunk_slice.data(),
                                                                     cute::flatten(output_chunk_slice.layout())),
                                                   cute::make_tensor(c_chunk.data(), cute::flatten(c_chunk.layout())));

                                        auto tmem_chunk =
                                            get_tmem_chunk(tmem_source,
                                                           cute::make_coord(chunk_inside_atom_block_index_m,
                                                                            chunk_inside_atom_block_index_n));

                                        CUTE_UNROLL
                                        for (int element_inside_chunk_index = 0;
                                             element_inside_chunk_index < cute::size(tmem_chunk);
                                             ++element_inside_chunk_index) {
                                            c_chunk(element_inside_chunk_index) = static_cast<d_engine_t>(
                                                lambda(tmem_chunk(element_inside_chunk_index),
                                                       c_chunk(element_inside_chunk_index),
                                                       coord_chunk(element_inside_chunk_index)));
                                        }

                                        cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits> {},
                                                   cute::make_tensor(c_chunk.data(), cute::flatten(c_chunk.layout())),
                                                   cute::make_tensor(output_chunk_slice.data(),
                                                                     cute::flatten(output_chunk_slice.layout())));
                                    }
                                }
                            } else {
                                // Direct (register) output IS the per-thread fragment,
                                // ordered like partition_like_C: N chunks within an M
                                // block first, then M blocks. Instead of computing linear
                                // fragment offsets, view the fragment as a 3D
                                // (element_inside_chunk, chunk_inside_tile_n, chunk_inside_tile_m)
                                // tensor - the column-major mode order reproduces exactly
                                // the fragment order (element fastest, then N, then M).
                                const auto chunk_elems = cute::size(cute::get<3>(tmem_source));
                                auto fragment_chunks   = cute::make_tensor(
                                    output.data(),
                                    cute::composition(output.layout(),
                                                      cute::make_layout(cute::make_shape(
                                                          chunk_elems,
                                                          cute::Int<atom_block_count_inside_tile_n> {} *
                                                              cute::size<4>(thread_tmem_chunks),
                                                          cute::Int<atom_block_count_inside_tile_m> {} *
                                                              cute::size<3>(thread_tmem_chunks)))));
                                // Per-element (m, n) coordinates for the user lambda,
                                // partitioned like the output tile (see the memory-backed
                                // branch above).
                                auto coord_chunks = cute::flat_divide(coord_tensor_t {}, tmem_chunk_tile_t {});
                                auto thread_coord_chunks = thread_tmem_to_register_copy.partition_D(coord_chunks);

                                CUTE_NO_UNROLL
                                for (int chunk_inside_atom_block_index_n = 0;
                                     chunk_inside_atom_block_index_n < cute::size<4>(thread_tmem_chunks);
                                     ++chunk_inside_atom_block_index_n) {
                                    CUTE_UNROLL
                                    for (int chunk_inside_atom_block_index_m = 0;
                                         chunk_inside_atom_block_index_m < cute::size<3>(thread_tmem_chunks);
                                         ++chunk_inside_atom_block_index_m) {
                                        const int chunk_inside_tile_index_m =
                                            atom_block_inside_tile_index_m * cute::size<3>(thread_tmem_chunks) +
                                            chunk_inside_atom_block_index_m;
                                        const int chunk_inside_tile_index_n =
                                            atom_block_inside_tile_index_n * cute::size<4>(thread_tmem_chunks) +
                                            chunk_inside_atom_block_index_n;
                                        auto tmem_chunk =
                                            get_tmem_chunk(tmem_source,
                                                           cute::make_coord(chunk_inside_atom_block_index_m,
                                                                            chunk_inside_atom_block_index_n));
                                        // Mode order differs between the two views:
                                        // fragment_chunks is (element, chunk_n, chunk_m),
                                        // thread_coord_chunks rest modes are (chunk_m, chunk_n).
                                        auto output_chunk = fragment_chunks(cute::_,
                                                                            chunk_inside_tile_index_n,
                                                                            chunk_inside_tile_index_m);
                                        auto coord_chunk  = thread_coord_chunks(cute::_,
                                                                               cute::_,
                                                                               cute::_,
                                                                               chunk_inside_tile_index_m,
                                                                               chunk_inside_tile_index_n);

                                        CUTE_UNROLL
                                        for (int element_inside_chunk_index = 0;
                                             element_inside_chunk_index < cute::size(tmem_chunk);
                                             ++element_inside_chunk_index) {
                                            output_chunk(element_inside_chunk_index) = static_cast<d_engine_t>(
                                                lambda(tmem_chunk(element_inside_chunk_index),
                                                       output_chunk(element_inside_chunk_index),
                                                       coord_chunk(element_inside_chunk_index)));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    epilogue_tmem_commit();
                } else {
                    static_assert(
                        is_supported_output,
                        "UTCMMA reduce_and_store output must be memory-backed or match the accumulator fragment size");
                }
            }

            // Accept mutable temporaries
            template<class OutEngine, class OutLayout, class Lambda>
            CUBLASDX_DEVICE void reduce_and_store(tensor<OutEngine, OutLayout>&& output,
                                                  Lambda&&                       lambda) const {
                reduce_and_store(output, static_cast<Lambda&&>(lambda));
            }

            // C = alpha * A * B + beta * C
            // Axpby can reuse reduce_and_store to avoid global memory spills from RMEM
            // It will be performed chunk by chunk
            template<class Alpha,
                     class Beta,
                     class ToEngine,
                     class ToLayout,
                     class LoadOp  = identity,
                     class StoreOp = identity>
            CUBLASDX_DEVICE void axpby(Alpha const&                alpha,
                                       Beta const&                 beta,
                                       tensor<ToEngine, ToLayout>& tD,
                                       LoadOp const&               load_op  = {},
                                       StoreOp const&              store_op = {}) const {
                assert(is_thread_active());
                using d_engine_t = typename ToEngine::value_type;

                auto cutlass_c_load_op  = transform_op_wrapper<LoadOp, d_engine_t> {load_op};
                auto cutlass_c_store_op = transform_op_wrapper<StoreOp, InputTypeC> {store_op};
                using cutlass_inout_value_type   = convert_to_cutlass_type_t<d_engine_t>;
                using cutlass_compute_value_type = convert_to_cutlass_type_t<InputTypeC>;

                reduce_and_store(tD, [&](auto acc, auto mem, auto const& /* coord */) -> d_engine_t {
                    auto c_val = static_cast<cutlass_compute_value_type>(
                        cutlass_c_load_op(static_cast<cutlass_inout_value_type>(mem)));
                    return static_cast<d_engine_t>(cutlass_c_store_op(
                        static_cast<cutlass_compute_value_type>(alpha) *
                            static_cast<cutlass_compute_value_type>(acc) +
                        static_cast<cutlass_compute_value_type>(beta) * c_val));
                });
            }

            template<class Alpha,
                     class Beta,
                     class ToEngine,
                     class ToLayout,
                     class LoadOp  = identity,
                     class StoreOp = identity>
            CUBLASDX_DEVICE void axpby(Alpha const&                 alpha,
                                       Beta const&                  beta,
                                       tensor<ToEngine, ToLayout>&& tD,
                                       LoadOp const&                load_op  = {},
                                       StoreOp const&               store_op = {}) const {
                axpby(alpha, beta, tD, load_op, store_op);
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_ACCUMULATOR_TMEM_ACCUMULATOR_HPP
