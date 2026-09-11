// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_RMEM_ACCUMULATOR_HPP
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_RMEM_ACCUMULATOR_HPP

#include "cublasdx/detail/accumulator/accumulator_helpers.hpp"
#include "cublasdx/detail/axpby.hpp"
#include "cublasdx/detail/blas_accumulator.hpp"
#include "cublasdx/detail/tensor.hpp"

namespace cublasdx {
    namespace detail {

        template<class InternalAccumulator, class ValueType>
        struct emulation_rmem_accumulator {
            using internal_accumulator_t = InternalAccumulator;
            using value_type             = ValueType;
            using register_fragment_t    = decltype(cute::make_fragment_like<value_type>(
                COMMONDX_STL_NAMESPACE::declval<typename internal_accumulator_t::register_fragment_t>()));
            static constexpr unsigned internal_alignment =
                cute::remove_cvref_t<decltype(internal_accumulator_t::alignment)>::value;

            unsigned            thread_idx;
            register_fragment_t accumulator_storage;

            CUBLASDX_DEVICE explicit emulation_rmem_accumulator(unsigned const in_thread_idx): thread_idx(in_thread_idx) {
                clear();
            }

            CUBLASDX_DEVICE auto mapping_accumulator() const {
                // TODO: benchmark against permanently holding as 
                // member (probably unnecessary slicing occurs inside)
                return internal_accumulator_t(thread_idx);
            }

            CUBLASDX_DEVICE void finish_accumulation() {}

            CUBLASDX_DEVICE void clear() {
                cute::clear(accumulator_storage);
            }

            CUBLASDX_DEVICE constexpr auto size() const {
                return cute::size(register_fragment_t {});
            }

            CUBLASDX_DEVICE auto is_thread_active() const {
                return mapping_accumulator().is_thread_active();
            }

            CUBLASDX_DEVICE auto is_predicated() const {
                return mapping_accumulator().is_predicated();
            }

            CUBLASDX_DEVICE auto get_alignment() const {
                return mapping_accumulator().get_alignment();
            }

            template<class CTensor>
            CUBLASDX_DEVICE auto partition_like_C(CTensor&& ctensor) const {
                return mapping_accumulator().partition_like_C(static_cast<CTensor&&>(ctensor));
            }

            template<class... Coords>
            CUBLASDX_DEVICE auto map_fragment_index(Coords&&... coords) const {
                return mapping_accumulator().map_fragment_index(static_cast<Coords&&>(coords)...);
            }

            template<class Coord>
            CUBLASDX_DEVICE auto is_coord_in_bounds(Coord const& coord) const {
                return mapping_accumulator().is_coord_in_bounds(coord);
            }

            template<class... Coords>
            CUBLASDX_DEVICE auto is_index_in_bounds(Coords&&... coords) const {
                return mapping_accumulator().is_index_in_bounds(static_cast<Coords&&>(coords)...);
            }

            template<class FromEngine, class FromLayout, class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_copy(tensor<FromEngine, FromLayout> const& tS,
                                                    tensor<ToEngine, ToLayout>&           tD) const {
                auto mapping = mapping_accumulator();
                ::cublasdx::detail::copy_fragment<internal_alignment>(tS, tD, mapping);
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
                auto frg = cute::make_fragment_like(partition_like_C(tS));
                auto mapping = mapping_accumulator();
                ::cublasdx::detail::copy_fragment<internal_alignment>(tS, frg, mapping);
                return frg;
            }

            CUBLASDX_DEVICE auto get_results() const {
                auto ret = cute::make_fragment_like(accumulator_storage);
                cute::copy(accumulator_storage, ret);
                return ret;
            }

            template<class Engine, class Layout>
            CUBLASDX_DEVICE void get_results(cublasdx::tensor<Engine, Layout>& out) const {
                cute::copy(accumulator_storage, out);
            }

            template<class Engine, class Layout>
            CUBLASDX_DEVICE void get_results(cublasdx::tensor<Engine, Layout>&& out) const {
                get_results(out);
            }

            template<class ToEngine, class ToLayout, class ProcessOp>
            CUBLASDX_DEVICE void process_and_store(tensor<ToEngine, ToLayout>& tD, ProcessOp&& process_op) const {
                using d_engine_t = typename ToEngine::value_type;
                auto acc_frag   = get_results();
                auto predicated = is_predicated();

                if constexpr (cute::is_rmem_v<ToEngine>) {
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(acc_frag); ++i) {
                        if constexpr (predicated) {
                            if (is_index_in_bounds(i)) {
                                tD(i) = static_cast<d_engine_t>(process_op(acc_frag(i)));
                            }
                        } else {
                            tD(i) = static_cast<d_engine_t>(process_op(acc_frag(i)));
                        }
                    }
                } else {
                    auto partitioned = partition_like_C(tD);
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(acc_frag); ++i) {
                        if constexpr (predicated) {
                            if (is_index_in_bounds(i)) {
                                partitioned(i) = static_cast<d_engine_t>(process_op(acc_frag(i)));
                            }
                        } else {
                            partitioned(i) = static_cast<d_engine_t>(process_op(acc_frag(i)));
                        }
                    }
                }
            }

            template<class ToEngine, class ToLayout, class ProcessOp>
            CUBLASDX_DEVICE void process_and_store(tensor<ToEngine, ToLayout>&& tD, ProcessOp&& process_op) const {
                process_and_store(tD, static_cast<ProcessOp&&>(process_op));
            }

            template<class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_store(tensor<ToEngine, ToLayout>& tD) const {
                process_and_store(tD, identity {});
            }

            template<class ToEngine, class ToLayout>
            CUBLASDX_DEVICE void partition_and_store(tensor<ToEngine, ToLayout>&& tD) const {
                partition_and_store(tD);
            }

            // The lambda is called as lambda(result_elem, output_elem, coord) where
            // coord is the element's (m, n) coordinate in the CTA output tile.
            // Out-of-bounds (padded) fragment slots are skipped and never handed
            // to the lambda.
            template<class OutEngine, class OutLayout, class Lambda>
            CUBLASDX_DEVICE void reduce_and_store(tensor<OutEngine, OutLayout>& output, Lambda&& lambda) const {
                using d_engine_t = typename OutEngine::value_type;
                auto acc_frag   = get_results();
                auto predicated = is_predicated();

                if constexpr (cute::is_rmem_v<OutEngine>) {
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(acc_frag); ++i) {
                        if constexpr (predicated) {
                            if (is_index_in_bounds(i)) {
                                output(i) =
                                    static_cast<d_engine_t>(lambda(acc_frag(i), output(i), map_fragment_index(i)));
                            }
                        } else {
                            output(i) = static_cast<d_engine_t>(lambda(acc_frag(i), output(i), map_fragment_index(i)));
                        }
                    }
                } else {
                    auto partitioned = partition_like_C(output);
                    CUTE_UNROLL
                    for (int i = 0; i < cute::size(acc_frag); ++i) {
                        if constexpr (predicated) {
                            if (is_index_in_bounds(i)) {
                                partitioned(i) = static_cast<d_engine_t>(
                                    lambda(acc_frag(i), partitioned(i), map_fragment_index(i)));
                            }
                        } else {
                            partitioned(i) =
                                static_cast<d_engine_t>(lambda(acc_frag(i), partitioned(i), map_fragment_index(i)));
                        }
                    }
                }
            }

            template<class OutEngine, class OutLayout, class Lambda>
            CUBLASDX_DEVICE void reduce_and_store(tensor<OutEngine, OutLayout>&& output, Lambda&& lambda) const {
                reduce_and_store(output, static_cast<Lambda&&>(lambda));
            }

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
                using d_engine_t = typename ToEngine::value_type;

                auto c_frag = get_results();

                auto cutlass_c_store_op          = transform_op_wrapper<StoreOp, value_type> {store_op};
                using cutlass_inout_value_type   = convert_to_cutlass_type_t<d_engine_t>;
                using cutlass_compute_value_type = convert_to_cutlass_type_t<value_type>;
                auto const& cutlass_alpha        = reinterpret_cast<cutlass_compute_value_type const&>(alpha);
                auto const& cutlass_beta         = reinterpret_cast<cutlass_compute_value_type const&>(beta);

                if (is_zero(cutlass_beta)) {
                    auto d_frag_io                    = cute::make_fragment_like(partition_like_C(tD));
                    auto cutlass_view_c_fragment      = cute::recast<cutlass_compute_value_type>(c_frag);
                    auto cutlass_view_inout_fragment_d = cute::recast<cutlass_inout_value_type>(d_frag_io);
                    cublasdx::transform_fragment(cutlass_view_c_fragment, cutlass_view_inout_fragment_d, [&](auto acc) {
                        cutlass_compute_value_type scaled_acc;
                        cublasdx::detail::mul(scaled_acc, cutlass_alpha, acc);
                        return cutlass_c_store_op(scaled_acc);
                    });
                    partition_and_copy(d_frag_io, tD);
                } else {
                    auto d_frag_io      = make_partition_and_copy(tD);
                    auto d_frag_compute = cute::make_fragment_like(c_frag);

                    auto cutlass_c_load_op             = transform_op_wrapper<LoadOp, d_engine_t> {load_op};
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

        template<class Accumulator>
        struct is_emulation_rmem_accumulator: cute::false_type {};

        template<class InternalAccumulator, class ValueType>
        struct is_emulation_rmem_accumulator<emulation_rmem_accumulator<InternalAccumulator, ValueType>>:
            cute::true_type {};

        template<class Accumulator>
        inline constexpr bool is_emulation_rmem_accumulator_v =
            is_emulation_rmem_accumulator<cute::remove_cvref_t<Accumulator>>::value;

        template<class InternalAccumulator, class ValueType>
        struct is_valid_blas_accumulator_impl<emulation_rmem_accumulator<InternalAccumulator, ValueType>>:
            cute::true_type {};

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_EMULATION_RMEM_ACCUMULATOR_HPP
