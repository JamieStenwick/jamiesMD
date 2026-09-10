// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_COPY_ATOMS_HPP
#define CUBLASDX_DETAIL_PIPELINE_COPY_ATOMS_HPP

#include "cublasdx/detail/pipeline/accumulator_mode.hpp"
#include "cublasdx/detail/copy.hpp"
#include <cute/atom/copy_traits_sm100_tma.hpp>

namespace cublasdx {

    namespace detail {

        // LDG+STS GMEM->SMEM copy lambda provider
        template<int Rank, int NumCopyThreads, int Alignment, class GmemDescriptor, class Tiler>
        struct sync_copy_gmem {
            static_assert(cute::is_static_v<Tiler>);

            static constexpr int rank = Rank;

            GmemDescriptor gmem_descriptor;

            CUBLASDX_PIPELINE_EXECUTION
            sync_copy_gmem(GmemDescriptor gmem): gmem_descriptor(gmem) {}

            template<class Crd>
            CUBLASDX_DEVICE auto partition(Crd const& crd) const {
                return cute::local_tile(gmem_descriptor, Tiler {}, crd);
            }

            CUBLASDX_DEVICE
            auto get_copy_functor() const {
                return [=](unsigned const tid, auto&& /* barrier */, auto const& gmem_tile, auto&& smem_tile) {
                    cublasdx::detail::copy_sync_impl<NumCopyThreads, Alignment>(tid, gmem_tile, smem_tile);
                };
            }
        };

        // LDGSTS GMEM->SMEM copy lambda provider
        template<int Rank, int NumCopyThreads, int Alignment, class GmemDescriptor, class Tiler>
        struct async_copy_gmem {
            static_assert(cute::is_static_v<Tiler>);

            static constexpr int rank = Rank;

            GmemDescriptor gmem_descriptor;

            CUBLASDX_PIPELINE_EXECUTION
            async_copy_gmem(GmemDescriptor gmem): gmem_descriptor(gmem) {}

            template<class Crd>
            CUBLASDX_DEVICE auto partition(Crd const& crd) const {
                return cute::local_tile(gmem_descriptor, Tiler {}, crd);
            }

            CUBLASDX_DEVICE
            auto get_copy_functor() const {
                return [&](unsigned const tid, auto&& /* barrier */, auto const& gmem_tile, auto&& smem_tile) {
                    cublasdx::detail::copy_impl<NumCopyThreads, Alignment>(tid, gmem_tile, smem_tile);
                };
            }
        };

        // UTMALDG GMEM->SMEM copy lambda provider
        template<int Rank, int NumCopyThreads, int Alignment, class TmaAtom, class Tiler>
        struct bulk_copy_gmem {
            static_assert(cute::is_static_v<Tiler>);
            static_assert(Alignment == 16);

            using shape_t = cute::
                conditional_t<Rank == 2, cute::Shape<unsigned, unsigned>, cute::Shape<unsigned, unsigned, unsigned>>;

            static constexpr int rank = Rank;

            TmaAtom tma_atom;
            shape_t shape;

            CUBLASDX_PIPELINE_EXECUTION
            bulk_copy_gmem(TmaAtom tma_atom, shape_t shape): tma_atom(tma_atom), shape(shape) {}

            template<class Crd>
            CUBLASDX_DEVICE auto partition(Crd const& crd) const {
                const auto coord_tensor = tma_atom.get_tma_tensor(shape);
                return cute::local_tile(coord_tensor, Tiler {}, crd);
            }

            CUBLASDX_DEVICE
            auto get_copy_functor() const {
                return [&](unsigned const tid, auto&& barrier, auto const& gmem_tile, auto&& smem_tile) {
                    auto& raw_barrier = detail::raw_barrier_storage(static_cast<decltype(barrier)&&>(barrier));
                    auto [gt, st] = tma_partition(tma_atom,
                                                  cute::Int<0> {},
                                                  cute::Layout<cute::_1> {},
                                                  cute::group_modes<0, 2>(smem_tile),
                                                  cute::group_modes<0, 2>(gmem_tile));
                    cute::copy(tma_atom.with(raw_barrier), gt, st);
                };
            }
        };

        // CopyAtom factory depending on chosen kernel strategy
        template<copy_kind CopyKind,
                 unsigned  NumCopyThreads,
                 unsigned  Alignment,
                 class GmemTensor,
                 class SmemLayout,
                 class Tiler>
        CUBLASDX_PIPELINE_EXECUTION constexpr auto make_gmem_descriptor(GmemTensor const& gmem,
                                                SmemLayout const& smem_layout,
                                                Tiler const&      tiler) {
            constexpr int rank = decltype(cute::rank(gmem.layout()))::value;

            if constexpr (CopyKind == copy_kind::sync) {
                return sync_copy_gmem<rank, NumCopyThreads, Alignment, GmemTensor, Tiler>(gmem);
            } else if constexpr (CopyKind == copy_kind::async) {
                return async_copy_gmem<rank, NumCopyThreads, Alignment, GmemTensor, Tiler>(gmem);
            } else {
                static_assert(CopyKind == copy_kind::bulk);
                using cutlass_value_type = convert_to_cutlass_type_t<typename GmemTensor::value_type>;
                auto const tma_atom =
                    CUBLASDX_MAKE_TMA_ATOM<cutlass_value_type>(cute::SM90_TMA_LOAD {}, gmem, smem_layout, tiler);
                using tma_atom_t = cute::remove_cvref_t<decltype(tma_atom)>;

                // Don't capture rank otherwise GCC8 would complain about
                // lambda capture of ‘rank’ is not a constant expression
                auto const gmem_shape = [&gmem]() {
                    if constexpr (rank == 2) {
                        return cute::make_shape(static_cast<unsigned>(cute::size<0>(gmem)),
                                                static_cast<unsigned>(cute::size<1>(gmem)));
                    } else {
                        return cute::make_shape(static_cast<unsigned>(cute::size<0>(gmem)),
                                                static_cast<unsigned>(cute::size<1>(gmem)),
                                                static_cast<unsigned>(cute::size<2>(gmem)));
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                return bulk_copy_gmem<rank, NumCopyThreads, Alignment, tma_atom_t, Tiler>(tma_atom, gmem_shape);
            }

            CUTE_GCC_UNREACHABLE;
        };
    } // namespace detail
} // namespace cublasdx
#endif // CUBLASDX_DETAIL_PIPELINE_COPY_ATOMS_HPP
