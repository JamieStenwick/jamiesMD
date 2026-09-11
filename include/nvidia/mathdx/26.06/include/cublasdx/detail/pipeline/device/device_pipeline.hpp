// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_DEVICE_PIPELINE_HPP
#define CUBLASDX_DETAIL_PIPELINE_DEVICE_PIPELINE_HPP

#include "cublasdx/detail/pipeline/accumulator_mode.hpp"
#include "cublasdx/detail/pipeline/device/device_pipeline_traits.hpp"
#include "cublasdx/detail/pipeline/tile/rmem/rmem_unified_pipeline.hpp"
#include "cublasdx/detail/pipeline/tile/rmem/rmem_specialized_pipeline.hpp"
#include "cublasdx/detail/pipeline/tile/tmem/tmem_unified_pipeline.hpp"
#include "cublasdx/detail/pipeline/tile/tmem/tmem_specialized_pipeline.hpp"

namespace cublasdx {
    namespace detail {
        template<unsigned       PipelineDepth,
                 result_storage ResultStorage,
                 bool           DisableWarpSpecialization,
                 copy_kind      CopyKind,
                 class BLAS,
                 class GmemDescriptorA,
                 class GmemDescriptorB,
                 class SmemLayoutA,
                 typename IOTypeA,
                 class SmemLayoutB,
                 typename IOTypeB>
        struct device_pipeline {
            static_assert(cute::is_layout<SmemLayoutA>::value, "SmemLayoutA must be a cute::Layout");
            static_assert(cute::is_layout<SmemLayoutB>::value, "SmemLayoutB must be a cute::Layout");

            // TODO: extend to generic dimensional
            // Currently only rank 2 and rank 3 tensors are allowed
            // extensions to higher dimensions are trivial, and should
            // follow the same scheme of (Semantic, Semantic, Extra1, Extra2 ..., ExtraN)
            // which means that M, N, K dimensions are always in the front, so that each
            // shared memory tile is of dimension (tile_m/n/k, tile_m/n/k, 1, 1, ...)
            static constexpr int rank_a = GmemDescriptorA::rank;
            static constexpr int rank_b = GmemDescriptorB::rank;
            static_assert(rank_a == rank_b, "Ranks of A and B descriptors must be the same");
            static constexpr int rank = rank_a; // checked that rank_a == rank_b above

            static constexpr auto is_cooperative = cute::C<CopyKind != copy_kind::bulk> {};
            static constexpr auto is_tma         = cute::C<CopyKind == copy_kind::bulk> {};

            // Either global tensor or global TMA coordinate tensor
            GmemDescriptorA gmem_descriptor_a;
            GmemDescriptorB gmem_descriptor_b;

            SmemLayoutA smem_layout_a;
            SmemLayoutB smem_layout_b;

            // Is fast path chosen
            static constexpr bool is_suggested =
                cute::is_same_v<cute::remove_cvref_t<decltype(BLAS::suggest_layout_smem_a().layout)>, SmemLayoutA> and
                cute::is_same_v<cute::remove_cvref_t<decltype(BLAS::suggest_layout_smem_b().layout)>, SmemLayoutB> and
                cute::is_same_v<typename BLAS::a_value_type, IOTypeA> and
                cute::is_same_v<typename BLAS::b_value_type, IOTypeB>;

        private:
            using traits =
                device_pipeline_traits<PipelineDepth, DisableWarpSpecialization, CopyKind, BLAS, is_suggested>;

            // Non WS TMEM
            static constexpr bool is_tmem_unified = traits::is_tmem_unified;
            // WS TMEM
            static constexpr bool is_tmem_specialized = traits::is_tmem_specialized;
            // WS RMEM (WGMMA or SuperMMA)
            static constexpr bool is_rmem_specialized = traits::is_rmem_specialized;

        public:
            // supermma, gmma or utcmma
            static constexpr auto mma_instruction_kind = traits::mma_instruction_kind;
            static constexpr bool is_warp_specialized = traits::is_warp_specialized;

            // extra warps used for TMA/MMA issuing in gmma/utcmma WS 
            static constexpr int additional_tma_and_mma_threads = traits::additional_tma_and_mma_threads;
            static constexpr int max_threads_per_block = traits::max_threads_per_block;
            static constexpr int first_extra_warp_idx = traits::first_extra_warp_idx;

            // 128 for TMA
            static constexpr unsigned alignment_a = traits::alignment_a;
            static constexpr unsigned alignment_b = traits::alignment_b;

            // How much data will be transferred async in each stage (only GEMM data)
            static constexpr unsigned stage_bytes_a = cute::size(SmemLayoutA {}) * sizeof(typename BLAS::a_value_type);
            static constexpr unsigned stage_bytes_b = cute::size(SmemLayoutB {}) * sizeof(typename BLAS::b_value_type);

            // Persistent SMEM storage, no streaming
            static constexpr unsigned barrier_bytes = traits::barrier_bytes;

            static constexpr int tile_m = size_of_v_m<BLAS>;
            static constexpr int tile_n = size_of_v_n<BLAS>;
            static constexpr int tile_k = size_of_v_k<BLAS>;

            static constexpr dim3 block_dim = traits::block_dim;

            CUBLASDX_DEVICE_PIPELINE_EXECUTION
            constexpr int stages() const {
                return PipelineDepth;
            }

            CUBLASDX_DEVICE_PIPELINE_EXECUTION
            constexpr dim3 get_block_dim() const {
                return block_dim;
            }

            CUBLASDX_DEVICE_PIPELINE_EXECUTION
            auto get_layout_smem_a() const {
                // Can't use products because dynamic layouts are not supported
                return cute::append<3>(smem_layout_a,
                                       cute::make_layout(cute::make_shape(cute::Int<PipelineDepth> {}),
                                                         cute::make_stride(cute::cosize(smem_layout_a))));
            }

            CUBLASDX_DEVICE_PIPELINE_EXECUTION
            auto get_layout_smem_b() const {
                // Can't use products because dynamic layouts are not supported
                return cute::append<3>(smem_layout_b,
                                       cute::make_layout(cute::make_shape(cute::Int<PipelineDepth> {}),
                                                         cute::make_stride(cute::cosize(smem_layout_b))));
            }

            // This only resets cursor / counter / coordinates for next iteration. No flushing / waiting / syncing
            template<class TilePipeline, class CoordA, class CoordB>
            CUBLASDX_DEVICE cute::enable_if_t<is_pipeline<cute::remove_cvref_t<TilePipeline>>::value>
            reset_tile(TilePipeline&& tile_pipeline, CoordA const& coord_a, CoordB const& coord_b) const {

                static constexpr int rank_coord_a = decltype(cute::rank(coord_a))::value;
                static constexpr int rank_coord_b = decltype(cute::rank(coord_b))::value;
                static_assert(rank_coord_a == rank_coord_b, "Coordinate ranks must match");
                // Rank decremented by 1 because one of the coordinates is always cublasdx::slice
                static_assert(rank_coord_a == rank - 1, "Reset coordinates ranks must match with init ranks");

                if constexpr (rank == 2) {
                    auto tile_row_a =
                        gmem_descriptor_a.partition(cute::make_coord(cute::get<0>(coord_a), cublasdx::slice));
                    auto tile_col_b =
                        gmem_descriptor_b.partition(cute::make_coord(cublasdx::slice, cute::get<0>(coord_b)));
                    tile_pipeline.reset_tile(tile_row_a, tile_col_b);
                } else if constexpr (rank == 3) {
                    auto tile_row_a = gmem_descriptor_a.partition(
                        cute::make_coord(cute::get<0>(coord_a), cublasdx::slice, cute::get<1>(coord_a)));
                    auto tile_col_b = gmem_descriptor_b.partition(
                        cute::make_coord(cublasdx::slice, cute::get<0>(coord_b), cute::get<1>(coord_b)));
                    tile_pipeline.reset_tile(tile_row_a, tile_col_b);
                } else {
                    static_assert(rank == 2 or rank == 3, "Incorrect pipeline rank");
                }
            }

            // Get tile for CTA indexed X, Y. Batch dimension possible (either 2D or 3D coordinate)
            template<class CoordA, class CoordB>
            CUBLASDX_DEVICE auto get_tile(void* smem, CoordA const& coord_a, CoordB const& coord_b) const {

                static constexpr int rank_coord_a = decltype(cute::rank(coord_a))::value;
                static constexpr int rank_coord_b = decltype(cute::rank(coord_b))::value;

                static_assert(rank_coord_a == rank_coord_b, "Coordinate ranks must match");
                // Rank decremented by 1 because one of the coordinates is always cublasdx::slice
                static_assert(rank_coord_a == rank - 1, "Reset coordinates ranks must match with init ranks");

                if constexpr (is_tma) {
                    cute::prefetch_tma_descriptor(gmem_descriptor_a.tma_atom.get_tma_descriptor());
                    cute::prefetch_tma_descriptor(gmem_descriptor_b.tma_atom.get_tma_descriptor());
                }

                // Lambda is inelegant --> consider conditional_return for coord only
                auto tile_row_a = [&]() {
                    if constexpr (rank == 2) {
                        return gmem_descriptor_a.partition(cute::make_coord(cute::get<0>(coord_a), cublasdx::slice));
                    } else if constexpr (rank == 3) {
                        return gmem_descriptor_a.partition(
                            cute::make_coord(cute::get<0>(coord_a), cublasdx::slice, cute::get<1>(coord_a)));
                    } else {
                        static_assert(rank == 2 or rank == 3, "Only ranks of 2 and 3 are allowed");
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                using tile_row_a_t = cute::remove_cvref_t<decltype(tile_row_a)>;

                // Pass copy functor instead of GMEM descriptor to ensure that
                // TMA descriptors always remain in device_pipeline and don't get copied
                auto copy_functor_a    = gmem_descriptor_a.get_copy_functor();
                using copy_functor_a_t = cute::remove_cvref_t<decltype(copy_functor_a)>;

                // Lambda is inelegant --> consider conditional_return for coord only
                auto tile_col_b = [&]() {
                    if constexpr (rank == 2) {
                        return gmem_descriptor_b.partition(cute::make_coord(cublasdx::slice, cute::get<0>(coord_b)));
                    } else if constexpr (rank == 3) {
                        return gmem_descriptor_b.partition(
                            cute::make_coord(cublasdx::slice, cute::get<0>(coord_b), cute::get<1>(coord_b)));
                    } else {
                        static_assert(rank == 2 or rank == 3, "Only ranks of 2 and 3 are allowed");
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                using tile_col_b_t = cute::remove_cvref_t<decltype(tile_col_b)>;

                // Pass copy functor instead of GMEM descriptor to ensure that
                // TMA descriptors always remain in device_pipeline and don't get copied
                auto copy_functor_b    = gmem_descriptor_b.get_copy_functor();
                using copy_functor_b_t = cute::remove_cvref_t<decltype(copy_functor_b)>;

                // Add dimension for stages
                auto smem_pipe_layout_a = get_layout_smem_a();
                auto smem_pipe_layout_b = get_layout_smem_b();

                using smem_pipe_layout_a_t = cute::remove_cvref_t<decltype(smem_pipe_layout_a)>;
                using smem_pipe_layout_b_t = cute::remove_cvref_t<decltype(smem_pipe_layout_b)>;

                // TODO: generalize to any dimension 
                auto const k_chunks = [&]() {
                    if constexpr (rank == 2) {
                        return cute::size<2>(tile_row_a.layout());
                    } else if constexpr (rank == 3) {
                        return cute::size<3>(tile_row_a.layout());
                    } else {
                        static_assert(rank == 2 or rank == 3, "Only ranks of 2 and 3 are allowed");
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                // Dispatch to appropriate tile: rmem/tmem and WS/non-WS
                if constexpr (is_tmem_specialized) {
                    return tmem_specialized_pipeline < PipelineDepth, ResultStorage, CopyKind, BLAS, is_suggested,
                           additional_tma_and_mma_threads,
                           tile_row_a_t, copy_functor_a_t, tile_col_b_t, copy_functor_b_t,
                           smem_pipe_layout_a_t, IOTypeA, smem_pipe_layout_b_t,
                           IOTypeB > (k_chunks,
                                      tile_row_a,
                                      copy_functor_a,
                                      tile_col_b,
                                      copy_functor_b,
                                      static_cast<byte*>(smem),
                                      smem_pipe_layout_a,
                                      smem_pipe_layout_b);
                } else if constexpr (is_tmem_unified) {
                    return tmem_unified_pipeline < PipelineDepth, ResultStorage, CopyKind, BLAS, is_suggested,
                           additional_tma_and_mma_threads,
                           tile_row_a_t, copy_functor_a_t, tile_col_b_t, copy_functor_b_t,
                           smem_pipe_layout_a_t, IOTypeA, smem_pipe_layout_b_t,
                           IOTypeB > (k_chunks,
                                      tile_row_a,
                                      copy_functor_a,
                                      tile_col_b,
                                      copy_functor_b,
                                      static_cast<byte*>(smem),
                                      smem_pipe_layout_a,
                                      smem_pipe_layout_b);
                } else if constexpr (is_rmem_specialized) {
                    return rmem_specialized_pipeline < PipelineDepth, ResultStorage, CopyKind, BLAS, is_suggested,
                           additional_tma_and_mma_threads,
                           first_extra_warp_idx, tile_row_a_t, copy_functor_a_t, tile_col_b_t, copy_functor_b_t,
                           smem_pipe_layout_a_t, IOTypeA, smem_pipe_layout_b_t,
                           IOTypeB > (k_chunks,
                                      tile_row_a,
                                      copy_functor_a,
                                      tile_col_b,
                                      copy_functor_b,
                                      static_cast<byte*>(smem),
                                      smem_pipe_layout_a,
                                      smem_pipe_layout_b);
                } else {
                    return rmem_unified_pipeline < PipelineDepth, ResultStorage, CopyKind, BLAS, is_suggested,
                           BLAS::max_threads_per_block,
                           first_extra_warp_idx, tile_row_a_t, copy_functor_a_t, tile_col_b_t, copy_functor_b_t,
                           smem_pipe_layout_a_t, IOTypeA, smem_pipe_layout_b_t,
                           IOTypeB > (k_chunks,
                                      tile_row_a,
                                      copy_functor_a,
                                      tile_col_b,
                                      copy_functor_b,
                                      static_cast<byte*>(smem),
                                      smem_pipe_layout_a,
                                      smem_pipe_layout_b);
                }
            }

            template<class GmemTensorA, class GmemTensorB>
            CUBLASDX_DEVICE_PIPELINE_EXECUTION device_pipeline(GmemTensorA const& gmem_tensor_a,
                                                               GmemTensorB const& gmem_tensor_b,
                                                               SmemLayoutA const& layout_a,
                                                               SmemLayoutB const& layout_b):
                gmem_descriptor_a(gmem_tensor_a),
                gmem_descriptor_b(gmem_tensor_b),
                smem_layout_a(layout_a),
                smem_layout_b(layout_b) {}

// Ifdef required by libmathdx to allow for device-initiated device_pipeline construction
#ifdef CUBLASDX_OVERLOAD_DEVICE_PIPELINE_CREATION
            CUBLASDX_DEVICE_PIPELINE_EXECUTION
            static constexpr auto buffer_alignment() {
                return alignment_a;
            }

            // Return size of entire shared memory buffer administered by tile_pipeline
            // stages + barriers + [tmem]
            CUBLASDX_DEVICE_PIPELINE_EXECUTION
            static constexpr unsigned buffer_size() {
                constexpr unsigned const shared_bytes_ab =
                    round_up(PipelineDepth * cute::cosize(SmemLayoutA {}) * sizeof(IOTypeA), alignment_b) +
                    PipelineDepth * cute::cosize(SmemLayoutB {}) * sizeof(IOTypeB);

                constexpr unsigned shared_bytes_base =
                    round_up(shared_bytes_ab, barrier_buffer_alignment_bytes) + barrier_bytes;

                return traits::shared_storage_size(shared_bytes_base);
            }
#else
            CUBLASDX_DEVICE_PIPELINE_EXECUTION
            constexpr auto buffer_alignment() const {
                return alignment_a;
            }

            // Return size of entire shared memory buffer administered by tile_pipeline
            // stages + barriers + [tmem]
            CUBLASDX_DEVICE_PIPELINE_EXECUTION
            unsigned buffer_size() const {
                unsigned const shared_bytes_ab =
                    round_up(PipelineDepth * cute::cosize(smem_layout_a) * sizeof(IOTypeA), alignment_b) +
                    PipelineDepth * cute::cosize(smem_layout_b) * sizeof(IOTypeB);

                unsigned shared_bytes = round_up(shared_bytes_ab, barrier_buffer_alignment_bytes) + barrier_bytes;

                return traits::shared_storage_size(shared_bytes);
            }
#endif
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_DEVICE_PIPELINE_HPP
