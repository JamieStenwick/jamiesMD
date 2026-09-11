// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_ACCUMULATOR_RMEM_ACCUMULATOR_HPP
#define CUBLASDX_DETAIL_ACCUMULATOR_RMEM_ACCUMULATOR_HPP

#include "cublasdx/detail/accumulator/accumulator_helpers.hpp"
#include "cublasdx/detail/axpby.hpp"
#include "cublasdx/detail/accumulator/copy_fragment.hpp"

namespace cublasdx {
    namespace detail {
        // Register memory accumulator, 
        // used with superMMA and WGMMA (Hopper)
        template<class TiledMMA,
                 instruction_type InstructionKind,
                 class LoadInstruction,
                 class StoreInstruction,
                 class ShapeMN,
                 class InputTypeC,
                 class Alignment,
                 class HasStaticBlockDim,
                 class BlockSize>
        struct rmem_accumulator {
            using atom_shape_mnk = typename TiledMMA::AtomShape_MNK;
            static constexpr Alignment         alignment            = {};
            static constexpr HasStaticBlockDim has_static_block_dim = {};

            // Type of ThrMMA
            using sliced_mma_t   = decltype(TiledMMA().get_thread_slice(cute::declval<unsigned>()));
            // Type used for index mapping
            using coord_tensor_t = decltype(cute::make_identity_tensor(ShapeMN {}));

            // True if threads are out of bounds
            static constexpr bool is_partition_divisible =
                decltype(cute::evenly_divides(ShapeMN {},
                                              cute::select<0, 1>(cute::tile_shape(TiledMMA {}))))::value;

            // Type of register backing memory tensor
            using register_fragment_t =
                decltype(cute::make_tensor<InputTypeC>(cute::partition_shape_C(sliced_mma_t {}, ShapeMN {})));
            using value_type = InputTypeC;

            TiledMMA tiled_mma = {};

            // Dynamic values
            unsigned     thr_idx;
            sliced_mma_t sliced_mma;

            // Storage
            register_fragment_t accumulator_storage;

            // Internal: Used by copy_fragment to get native copy mechanism
            // exposes heuristic constructed copy atom
            CUBLASDX_DEVICE
            auto make_tiled_load_copy_c() const {
                auto tiled_copy = cute::make_tiled_copy_C(cute::Copy_Atom<LoadInstruction, InputTypeC> {}, sliced_mma);
                auto thr_copy   = tiled_copy.get_thread_slice(thr_idx);
                return cute::make_tuple(tiled_copy, thr_copy);
            }

            // Internal: Used by copy_fragment to get native_copy_mechanism
            // exposes heuristic constructed copy atom
            CUBLASDX_DEVICE
            auto make_tiled_store_copy_c() const {
                auto tiled_copy = cute::make_tiled_copy_C(cute::Copy_Atom<StoreInstruction, InputTypeC> {}, sliced_mma);
                auto thr_copy   = tiled_copy.get_thread_slice(thr_idx);
                return cute::make_tuple(tiled_copy, thr_copy);
            }

            CUBLASDX_DEVICE
            void turn_on_accumulation() {
                static_assert(InstructionKind == instruction_type::gmma, "This functionality is only available for WGMMA");
                tiled_mma.accumulate_ = cute::SM90::GMMA::ScaleOut::One;
            }

            CUBLASDX_DEVICE
            void turn_off_accumulation() {
                static_assert(InstructionKind == instruction_type::gmma, "This functionality is only available for WGMMA");
                tiled_mma.accumulate_ = cute::SM90::GMMA::ScaleOut::Zero;
            }

            public:
            // NOP kept for keeping the same interface with TMEM staging accumulator
            CUBLASDX_DEVICE
            void finish_accumulation() {}

            // Always clear for next MMA in construction
            CUBLASDX_DEVICE
            rmem_accumulator(unsigned const thread_idx):
                thr_idx(thread_idx),
                sliced_mma(TiledMMA().get_slice(thread_idx)) {
                // This will either clear registers (superMMA) or set MMA flag to zero
                // for WGMMA (it performs C = A * B instead of C += A * B then)
                clear();
            }

            CUBLASDX_DEVICE
            constexpr auto is_predicated() const {
                // Not using C<bool> for GCC7 sanity
                return cute::conditional_return<is_partition_divisible>(cute::false_type {}, cute::true_type {});
            }

            CUBLASDX_DEVICE
            constexpr auto get_alignment() const { return alignment; }

            // Clear accumulator either through zero-setting or resetting accumulation
            // bit on GMMA accumulator
            CUBLASDX_DEVICE
            void clear() { 
                if constexpr(InstructionKind == instruction_type::gmma) {
                    turn_off_accumulation();
                } else {
                    cute::clear(accumulator_storage); 
                }
            }

            CUBLASDX_DEVICE
            constexpr auto size() const { return cute::size(register_fragment_t {}); }

            // True if this thread takes part in accumulator element partitioning
            // Can be used for warp-specialized detection as well
            CUBLASDX_DEVICE
            auto is_thread_active() const {
                // Always when GMMA is used return actual value without
                // short-circuiting. This allows nvmath-python to avoid 
                // using lambdas for thread divisions
                // IMPLICIT: SuperMMA is always unified
                if constexpr (InstructionKind == instruction_type::gmma) {
                    static_assert(BlockSize::value % 128 == 0, "WGMMA can be used only with full warpgroups");
                    return thr_idx < BlockSize::value;
                } else if constexpr (has_static_block_dim) {
                    // Otherwise if static block dim specified, shortcircuit
                    return cute::true_type {};
                } else {
                    // Functionally the same as BlockSize::value
                    // with forward compatibility for MMA thread 
                    // sizes smaller than provided threads
                    return thr_idx < cute::size(tiled_mma);
                }
                CUTE_GCC_UNREACHABLE;
            }

            // Use TiledMMA and ThrMMA to "choose" (partition) only inout elements
            // that will belong to the current thread
            // Input: SMEM/GMEM tensor of (SizeM, SizeN --> from Size<M, N, K>)
            // Output: slice of input tensor (ThreadM, ThreadN) with available
            // elements only belonging to current thread
            template<class CTensor>
            CUBLASDX_DEVICE auto partition_like_C(CTensor&& ctensor) const {
                return sliced_mma.partition_C(static_cast<CTensor&&>(ctensor));
            }

            // Maps arbitrary local fragment index to CTA matrix index
            // Index-space inverse of partition_like_C
            template<class... Coords>
            CUBLASDX_DEVICE auto map_fragment_index(Coords&&... coords) const {
                auto thr_coord = partition_like_C(coord_tensor_t {});
                return thr_coord(static_cast<Coords&&>(coords)...);
            }

            // Checks if an (m, n) coordinate lies inside the (M, N) problem shape.
            // Companion to the reduce_and_store lambda contract: padded fragment
            // slots of a predicated GEMM carry coordinates outside this shape.
            template<class Coord>
            CUBLASDX_DEVICE auto is_coord_in_bounds(Coord const& coord) const {
                return cute::elem_less(coord, ShapeMN {});
            }

            // Checks if **local** fragment index is in bounds for storing to CTA-wide
            // matrix. Some threads will have all elements in bounds, while some don't
            template<class... Coords>
            CUBLASDX_DEVICE auto is_index_in_bounds(Coords&&... coords) const {
                // If not predicated then short-circuit
                if constexpr (not is_partition_divisible) {
                    return is_coord_in_bounds(map_fragment_index(static_cast<Coords&&>(coords)...));
                } else {
                    return cute::true_type{};
                }

                CUTE_GCC_UNREACHABLE;
            }

            // Make empty register tensor of shape exactly like output fragment
            // clear it by setting elements to 0 
            CUBLASDX_DEVICE
            auto make_empty_fragment() const {
                auto ret = cute::make_fragment_like(register_fragment_t {});
                cute::clear(ret);
                return ret;
            }

            // 1. Detect either input or output as (SizeM, SizeN) CTA-wide tensor
            // 2. Partition that 
            // 3. Store this thread's elements into appropriate partition
            template<class FromEngine, class FromLayout, class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_copy(tensor<FromEngine, FromLayout> const& tS,
                                                    tensor<ToEngine, ToLayout>&           tD) const {
                assert(is_thread_active());
                ::cublasdx::detail::copy_fragment<Alignment::value>(tS, tD, *this);
            }

            // Accept mutable temporaries
            template<class FromEngine, class FromLayout, class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_copy(tensor<FromEngine, FromLayout> const& tS,
                                                    tensor<ToEngine, ToLayout>&&          tD) const {
                partition_and_copy(tS, tD);
            }

            // 1. Construct local register fragment
            // 2. Partition CTA-wide tensor
            // 3. Load thread-appropriate elements into that tensor
            // 4. Return tensor
            template<class FromEngine,
                     class FromLayout,
                     cute::enable_if_t<cute::is_smem_v<FromEngine> or cute::is_gmem_v<FromEngine>>* = nullptr>
            CUBLASDX_DEVICE auto make_partition_and_copy(tensor<FromEngine, FromLayout> const& tS) const {
                assert(is_thread_active());
                auto frg = cute::make_fragment_like(partition_like_C(tS));
                ::cublasdx::detail::copy_fragment<Alignment::value>(tS, frg, *this);
                return frg;
            }

            // A new fragment is created to avoid ownership issues 
            CUBLASDX_DEVICE
            auto get_results() const {
                assert(is_thread_active());
                auto ret = cute::make_fragment_like(accumulator_storage);
                cute::copy(accumulator_storage, ret);
                return ret;
            }

            // Copy results into user-provided tensor
            template<class Engine, class Layout>
            CUBLASDX_DEVICE
            void get_results(cublasdx::tensor<Engine, Layout>& out) const {
                assert(is_thread_active());
                cute::copy(accumulator_storage, out);
            }

            // Accept mutable temporaries
            template<class Engine, class Layout>
            CUBLASDX_DEVICE
            void get_results(cublasdx::tensor<Engine, Layout>&& out) const {
                get_results(out);
            }

            // 1. Partition CTA-wide tensor
            // 2. Extract results and store into thread-appropriate storage
            template<class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_store(tensor<ToEngine, ToLayout>& tD) const {
                assert(is_thread_active());
                auto results = get_results();
                ::cublasdx::detail::copy_fragment<Alignment::value>(results, tD, *this);
            }

            // Accept mutable temporaries
            template<class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_store(tensor<ToEngine, ToLayout>&& tD) const {
                partition_and_store(tD);
            }

            // Apply per-element transform to results and then store them to provided CTA-tensor
            // Why? For TMEM tensor compatibility.
            // TODO: Interleave ops and stores
            template<class ToEngine, class ToLayout, class ProcessOp>
            CUBLASDX_DEVICE void process_and_store(tensor<ToEngine, ToLayout>& tD,
                                                   ProcessOp&&                 process_op) const {
                assert(is_thread_active());
                using d_engine_t = typename ToEngine::value_type;
                auto acc_frag = get_results();
                auto partitioned = partition_like_C(tD);

                CUTE_UNROLL
                for (int i = 0; i < cute::size(acc_frag); ++i) {
                    if constexpr (!is_partition_divisible) {
                        if (is_index_in_bounds(i)) {
                            partitioned(i) = static_cast<d_engine_t>(process_op(acc_frag(i)));
                        }
                    } else {
                        partitioned(i) = static_cast<d_engine_t>(process_op(acc_frag(i)));
                    }
                }
            }

            // Accept mutable temporaries
            template<class ToEngine, class ToLayout, class ProcessOp>
            CUBLASDX_DEVICE void process_and_store(tensor<ToEngine, ToLayout>&& tD,
                                                   ProcessOp&&                  process_op) const {
                process_and_store(tD, static_cast<ProcessOp&&>(process_op));
            }

            // 1. (optional) Partition provided CTA-wide tensor
            // 2. Load thread-appropriate partition of it
            // 3. For each pair of elements (Result, Output) perform the lambda
            //    lambda(result_elem, output_elem, coord) where coord is the
            //    element's (m, n) coordinate in the CTA output tile (the same
            //    value map_fragment_index returns for its fragment index)
            // 4. Store function output back to tensor
            // For CTA-wide memory outputs, out-of-bounds (padded) fragment
            // slots are skipped and never handed to the lambda. A register
            // output IS the fragment, so the lambda runs over all of it -
            // including padded slots, whose coordinates lie outside (M, N).
            // TODO: Interleave loads/ops/stores
            // TODO: vectorize loads
            template<class OutEngine, class OutLayout, class Lambda>
            CUBLASDX_DEVICE void reduce_and_store(tensor<OutEngine, OutLayout>& output,
                                                  Lambda&&                      lambda) const {
                assert(is_thread_active());
                using d_engine_t = typename OutEngine::value_type;
                auto acc_frag  = get_results();
                auto thr_coord = partition_like_C(coord_tensor_t {});

                // Special case for doing a local reduction with RMEM tensor
                if constexpr (cute::is_rmem_v<OutEngine>) {
                    static_assert(cute::size(OutLayout {}) == cute::size(register_fragment_t {}),
                                  "register-backed reduce_and_store output must match the accumulator fragment size");
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(acc_frag); ++i) {
                        output(i) = static_cast<d_engine_t>(lambda(acc_frag(i), output(i), thr_coord(i)));
                    }
                } else {
                    auto partitioned = partition_like_C(output);
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(acc_frag); ++i) {
                        if constexpr (!is_partition_divisible) {
                            if (is_index_in_bounds(i)) {
                                auto out = static_cast<d_engine_t>(
                                    lambda(acc_frag(i), partitioned(i), thr_coord(i)));
                                partitioned(i) = out;
                            }
                        } else {
                            auto out = static_cast<d_engine_t>(
                                lambda(acc_frag(i), partitioned(i), thr_coord(i)));
                            partitioned(i) = out;
                        }
                    }
                }
            }

            // Accept mutable temporaries
            template<class OutEngine, class OutLayout, class Lambda>
            CUBLASDX_DEVICE void reduce_and_store(tensor<OutEngine, OutLayout>&& output,
                                                  Lambda&&                       lambda) const {
                reduce_and_store(output, static_cast<Lambda&&>(lambda));
            }

            // 1. Take CTA-wide C tensor as tensor input 
            // 2. Partition it into thread-appropriate slices
            // 3. Load entire partition
            // 4. Perform load_op lambda
            // 5. Perform axpby(alpha, input, beta, results)
            // 6. Apply store_op
            // 7. Store back to CTA-wide C tensor
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

                auto c_frag = get_results();

                auto cutlass_c_store_op          = transform_op_wrapper<StoreOp, InputTypeC> {store_op};
                using cutlass_inout_value_type   = convert_to_cutlass_type_t<d_engine_t>;
                using cutlass_compute_value_type = convert_to_cutlass_type_t<InputTypeC>;
                auto const& cutlass_alpha       = reinterpret_cast<cutlass_compute_value_type const&>(alpha);
                auto const& cutlass_beta        = reinterpret_cast<cutlass_compute_value_type const&>(beta);

                if (is_zero(cutlass_beta)) {
                    auto d_frag_io                      = cute::make_fragment_like(partition_like_C(tD));
                    auto cutlass_view_c_fragment        = cute::recast<cutlass_compute_value_type>(c_frag);
                    auto cutlass_view_inout_fragment_d   = cute::recast<cutlass_inout_value_type>(d_frag_io);
                    cublasdx::transform_fragment(cutlass_view_c_fragment, cutlass_view_inout_fragment_d, [&](auto acc) {
                        cutlass_compute_value_type scaled_acc;
                        cublasdx::detail::mul(scaled_acc, cutlass_alpha, acc);
                        return cutlass_c_store_op(scaled_acc);
                    });
                    partition_and_copy(d_frag_io, tD);
                } else {
                    auto d_frag_io      = make_partition_and_copy(tD);
                    auto d_frag_compute = cute::make_fragment_like(c_frag);

                    auto cutlass_c_load_op               = transform_op_wrapper<LoadOp, d_engine_t> {load_op};
                    auto cutlass_view_compute_fragment_d = cute::recast<cutlass_compute_value_type>(d_frag_compute);
                    auto cutlass_view_inout_fragment_d   = cute::recast<cutlass_inout_value_type>(d_frag_io);

                    cublasdx::transform_fragment(
                        cutlass_view_inout_fragment_d, cutlass_view_compute_fragment_d, cutlass_c_load_op);
                    ::cublasdx::axpby(alpha, c_frag, beta, d_frag_compute);
                    cublasdx::transform_fragment(
                        cutlass_view_compute_fragment_d, cutlass_view_inout_fragment_d, cutlass_c_store_op);
                    partition_and_copy(d_frag_io, tD);
                }
            }

            // Accept mutable temporaries
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

#endif // CUBLASDX_DETAIL_ACCUMULATOR_RMEM_ACCUMULATOR_HPP
