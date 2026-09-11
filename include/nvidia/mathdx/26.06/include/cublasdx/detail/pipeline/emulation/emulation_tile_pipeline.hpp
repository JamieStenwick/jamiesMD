// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_TILE_PIPELINE_HPP
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_TILE_PIPELINE_HPP

#include "cublasdx/detail/pipeline/emulation/emulation_rmem_accumulator.hpp"
#include "cublasdx/detail/pipeline/emulation/exponents.hpp"
#include "cublasdx/traits.hpp"

namespace cublasdx {
    namespace detail {

        template<class ExternalBLAS,
                 class InternalBLAS,
                 class InternalDevicePipeline,
                 class InternalTilePipeline,
                 class AShiftSmemTensor,
                 class BShiftSmemTensor,
                 int Slices>
        struct emulation_tile_pipeline {
            using internal_tile_pipeline_t = InternalTilePipeline;
            using external_blas_t          = ExternalBLAS;
            using internal_blas_t          = InternalBLAS;
            using internal_accumulator_t   = cute::remove_cvref_t<decltype(
                COMMONDX_STL_NAMESPACE::declval<InternalTilePipeline&>().get_accumulator())>;
            using accumulator_t = emulation_rmem_accumulator<internal_accumulator_t, typename ExternalBLAS::c_value_type>;
            static constexpr int max_threads_per_block = InternalDevicePipeline::max_threads_per_block;

            InternalTilePipeline          internal_tile_pipeline;
            // Persistent internal accumulator: its TMEM stage/phase state must carry
            // across emulated tiles (like the direct GEMM's this->internal_accumulator())
            // so it stays in lockstep with the compute/epilogue TMEM barriers, which
            // reset_tile does not re-initialize. Re-minting it per tile (producer_start)
            // desyncs the barrier phase and deadlocks on tile reuse.
            internal_accumulator_t        int_accumulator;
            InternalDevicePipeline const* internal_device_pipeline_ptr = nullptr;
            AShiftSmemTensor             smem_shift_a;
            BShiftSmemTensor             smem_shift_b;
            unsigned                     output_tile_m = 0;
            unsigned                     output_tile_n = 0;
            int                          slice_count = Slices;

            template<class Smem, class CoordA, class CoordB>
            CUBLASDX_DEVICE emulation_tile_pipeline(InternalDevicePipeline const& internal_device_pipeline,
                                                    Smem                          pipeline_smem,
                                                    CoordA const&                 coord_a,
                                                    CoordB const&                 coord_b,
                                                    AShiftSmemTensor const&       in_smem_shift_a,
                                                    BShiftSmemTensor const&       in_smem_shift_b,
                                                    unsigned const                tile_m,
                                                    unsigned const                tile_n,
                                                    int const                     in_slice_count):
                internal_tile_pipeline(internal_device_pipeline.get_tile(pipeline_smem, coord_a, coord_b)),
                int_accumulator(internal_tile_pipeline.get_accumulator()),
                internal_device_pipeline_ptr(&internal_device_pipeline),
                smem_shift_a(in_smem_shift_a),
                smem_shift_b(in_smem_shift_b),
                output_tile_m(tile_m),
                output_tile_n(tile_n),
                slice_count(in_slice_count) {}

            CUBLASDX_DEVICE auto get_accumulator() {
                return accumulator_t(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
            }

            CUBLASDX_DEVICE void epilogue_sync() {
                internal_tile_pipeline.epilogue_sync();
            }

        private:
            CUBLASDX_DEVICE InternalDevicePipeline const& internal_device_pipeline() const {
                return *internal_device_pipeline_ptr;
            }

            CUBLASDX_DEVICE void reset_internal_tile(int const a_slice, int const b_slice) {
                internal_device_pipeline().reset_tile(
                    internal_tile_pipeline,
                    cublasdx::make_coord(output_tile_m, a_slice),
                    cublasdx::make_coord(output_tile_n, b_slice));
            }

            template<class Fp64Accumulator, class IntAccumulator>
            CUBLASDX_DEVICE void accumulate_diagonal(int const diag,
                                                     Fp64Accumulator& fp64_accumulator,
                                                     IntAccumulator const& int_accumulator) {
                // reduce_and_store hands the lambda the element's (m, n) tile
                // coordinate; for a register output it iterates the whole
                // fragment, so padded slots of a predicated GEMM arrive with
                // coordinates outside the tile shape and must not index the
                // shift tiles.
                int_accumulator.reduce_and_store(
                    fp64_accumulator.accumulator_storage,
                    [&](auto const& accumulator_elem, auto const& fp64_elem, auto const& coord) {
                        using fp64_value_t = typename Fp64Accumulator::value_type;
                        if constexpr (decltype(int_accumulator.is_predicated())::value) {
                            if (not int_accumulator.is_coord_in_bounds(coord)) {
                                // Padded fragment slots are not stored, but must not index shift tiles.
                                return fp64_value_t {0};
                            }
                        }

                        const auto shift_a_elem = smem_shift_a(cute::get<0>(coord));
                        const auto shift_b_elem = smem_shift_b(cute::get<1>(coord));
                        return static_cast<fp64_value_t>(fp64_elem) +
                               emulation_nth_slice_to_fp64<typename IntAccumulator::value_type,
                                                           typename InternalBLAS::a_value_type>(
                                   diag, accumulator_elem, shift_a_elem + shift_b_elem);
                    });
            }

        public:
            template<class Fp64Accumulator>
            CUBLASDX_DEVICE cute::enable_if_t<is_emulation_rmem_accumulator_v<Fp64Accumulator>>
            execute(Fp64Accumulator& fp64_accumulator) {
                // int_accumulator is a persistent member (see declaration): its stage/phase
                // carries across emulated tiles so it stays in sync with the TMEM barriers.
                for (int diag = slice_count - 1; diag >= 0; --diag) {
                    int_accumulator.clear();

                    for (int term = 0; term <= diag; ++term) {
                        internal_tile_pipeline.execute(int_accumulator);

                        if (term < diag) {
                            reset_internal_tile(term + 1, diag - (term + 1));
                        } else if (diag > 0) {
                            reset_internal_tile(0, diag - 1);
                        }
                    }

                    internal_tile_pipeline.epilogue(int_accumulator, [&](auto const& accumulator) {
                        accumulate_diagonal(diag, fp64_accumulator, accumulator);
                    });
                }
            }

            template<class BlasAccumulator, class EpilogueFunctor>
            CUBLASDX_DEVICE cute::enable_if_t<is_emulation_rmem_accumulator_v<BlasAccumulator>>
            epilogue(BlasAccumulator&& accumulator, EpilogueFunctor const& epilogue_functor) {
                accumulator.finish_accumulation();
                if (accumulator.is_thread_active()) {
                    epilogue_functor(accumulator);
                }
            }

            template<class EpilogueFunctor>
            CUBLASDX_DEVICE cute::enable_if_t<not is_valid_blas_accumulator_v<EpilogueFunctor>>
            execute(EpilogueFunctor&& epilogue_functor) {
                auto accumulator = get_accumulator();
                execute(accumulator);
                epilogue(accumulator, epilogue_functor);
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_EMULATION_TILE_PIPELINE_HPP
