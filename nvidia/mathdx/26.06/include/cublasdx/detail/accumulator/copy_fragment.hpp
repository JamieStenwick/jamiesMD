// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_ACCUMULATOR_COPY_FRAGMENT_HPP
#define CUBLASDX_DETAIL_ACCUMULATOR_COPY_FRAGMENT_HPP

#include "cublasdx/detail/tensor.hpp"

namespace cublasdx {
    namespace detail {

    // Copy fragment from registers to GMEM/SMEM
    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE COMMONDX_STL_NAMESPACE::enable_if_t<cute::is_rmem_v<TRC> and
                                                        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TRC, CFragLayout> const& tS, tensor<TC, CLayout>& tD, Partitioner const& p) {
        auto tPtD = p.partition_like_C(tD);

        using src_type               = typename TRC::value_type;
        using dst_type               = typename TC::value_type;
        using partitioner_value_type = typename Partitioner::value_type;

        // Check if copies like STSM are available (dependent on the preceding GEMM)
        constexpr bool GEMM_compliant_copy =
            cute::is_same_v<src_type, partitioner_value_type> and
            cute::is_same_v<dst_type, partitioner_value_type> and cute::is_smem_v<TC> and
            cute::is_rmem_v<TRC> and AlignmentInBytes == decltype(p.get_alignment())::value and
            ((cute::max_alignment(CLayout {}) * sizeof(typename TC::value_type)) ==
             decltype(p.get_alignment())::value);

        using src_shape = decltype(tS.shape());
        using dst_shape = decltype(tPtD.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
                      "cublasdx::copy requires static tensor layouts");

        auto predicated = p.is_predicated();

        // If the copy is predicated the best we can do is trivial conditional store
        if constexpr (predicated) {
            cute::copy_if(cute::lazy::transform(cute::make_identity_tensor(cute::shape(p.make_empty_fragment())),
                                                [&](auto idx) { return p.is_index_in_bounds(idx); }),
                          tS,
                          tPtD);
        } else 
        // If GEMM_compliant_copy is available then we use pre-computed copy atom with proper CuTe tiled_copy pattern
        if constexpr (GEMM_compliant_copy) {
            auto [tiled_copy, thr_copy] = p.make_tiled_store_copy_c();
            cute::Tensor tCsD           = thr_copy.partition_D(tD);
            // Virtually re-partition for copy access
            cute::Tensor tCrS_copy_view = thr_copy.retile_S(tS);
            CUTE_STATIC_ASSERT_V(cute::size<1>(tCsD) == cute::size<1>(tCrS_copy_view));
            CUTE_STATIC_ASSERT_V(cute::size<2>(tCsD) == cute::size<2>(tCrS_copy_view));
            copy(tiled_copy, tCrS_copy_view, tCsD);
        } 
        // otherwise just try to infer the widest STS/STG available 
        else {
            constexpr int max_vec_bits = cute::gcd(
                AlignmentInBytes * 8, cute::max_alignment(CLayout {}) * cute::sizeof_bits_v<dst_type>);
            using tile_m_dimension = decltype(cute::select<1>(tPtD.layout()));
            using tile_n_dimension = decltype(cute::select<2>(tPtD.layout()));
            // Always permute for static dimension to be present in the last dimension
            if constexpr (not cute::is_static<tile_m_dimension>::value and
                          cute::is_static<tile_n_dimension>::value) {
                const auto permuted_ts =
                    cute::make_tensor(tS.data(), cute::flatten(cute::select<0, 2, 1>(tS.layout())));
                auto permuted_tptd =
                    cute::make_tensor(tPtD.data(), cute::flatten(cute::select<0, 2, 1>(tPtD.layout())));
                cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits> {},
                           permuted_ts,
                           permuted_tptd);
            } else {
                const auto flat_ts   = cute::make_tensor(tS.data(), cute::flatten(tS.layout()));
                auto       flat_tptd = cute::make_tensor(tPtD.data(), cute::flatten(tPtD.layout()));
                cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits> {}, flat_ts, flat_tptd);
            }
        }
    }

    // Accept mutable temporaries
    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE COMMONDX_STL_NAMESPACE::enable_if_t<cute::is_rmem_v<TRC> and
                                                        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TRC, CFragLayout> const& tS, tensor<TC, CLayout>&& tD, Partitioner const& p) {
        copy_fragment<AlignmentInBytes>(tS, tD, p);
    }

    // Mirror function to the first one in this file, this time loading from GMEM/SMEM to registers
    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE COMMONDX_STL_NAMESPACE::enable_if_t<cute::is_rmem_v<TRC> and
                                                        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TC, CLayout> const& tS, tensor<TRC, CFragLayout>& tD, Partitioner const& p) {
        auto tPtS = p.partition_like_C(tS);

        using src_type               = typename TC::value_type;
        using dst_type               = typename TRC::value_type;
        using partitioner_value_type = typename Partitioner::value_type;

        // Check if copies like LDSM are available (dependent on previous GEMM)
        constexpr bool GEMM_compliant_copy =
            cute::is_same_v<src_type, partitioner_value_type> and
            cute::is_same_v<dst_type, partitioner_value_type> and cute::is_smem_v<TC> and
            cute::is_rmem_v<TRC> and AlignmentInBytes == decltype(p.get_alignment())::value and
            ((cute::max_alignment(CLayout {}) * sizeof(typename TC::value_type)) ==
             decltype(p.get_alignment())::value);

        using src_shape = decltype(tS.shape());
        using dst_shape = decltype(tPtS.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
                      "cublasdx::copy requires static tensor layouts");

        auto predicated = p.is_predicated();

        // If the copy is predicated the best we can do is trivial conditional store
        if constexpr (predicated) {
            cute::copy_if(cute::lazy::transform(cute::make_identity_tensor(cute::shape(p.make_empty_fragment())),
                                                [&](auto idx) { return p.is_index_in_bounds(idx); }),
                          tPtS,
                          tD);
        } else 
        // If GEMM_compliant_copy is available then we use pre-computed copy atom with proper CuTe tiled_copy pattern
        if constexpr (GEMM_compliant_copy) {
            auto [tiled_copy, thr_copy] = p.make_tiled_load_copy_c();
            cute::Tensor tCsS           = thr_copy.partition_S(tS);
            cute::Tensor tCrD_copy_view = thr_copy.retile_D(tD);
            CUTE_STATIC_ASSERT_V(cute::size<1>(tCsS) == cute::size<1>(tCrD_copy_view));
            CUTE_STATIC_ASSERT_V(cute::size<2>(tCsS) == cute::size<2>(tCrD_copy_view));
            copy(tiled_copy, tCsS, tCrD_copy_view);
        } 
        // otherwise just try to infer the widest STS/STG available 
        else {
            constexpr int max_vec_bits = cute::gcd(
                AlignmentInBytes * 8, cute::max_alignment(CLayout {}) * cute::sizeof_bits_v<src_type>);
            using tile_m_dimension = decltype(cute::select<1>(tPtS.layout()));
            using tile_n_dimension = decltype(cute::select<2>(tPtS.layout()));
            
            // Always permute for static dimension to be present in the last dimension
            if constexpr (not cute::is_static<tile_m_dimension>::value and
                          cute::is_static<tile_n_dimension>::value) {
                const auto permuted_tpts =
                    cute::make_tensor(tPtS.data(), cute::flatten(cute::select<0, 2, 1>(tPtS.layout())));
                auto permuted_td =
                    cute::make_tensor(tD.data(), cute::flatten(cute::select<0, 2, 1>(tD.layout())));
                cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits> {},
                           permuted_tpts,
                           permuted_td);
            } else {
                const auto flat_tpts = cute::make_tensor(tPtS.data(), cute::flatten(tPtS.layout()));
                auto       flat_td   = cute::make_tensor(tD.data(), cute::flatten(tD.layout()));
                cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<max_vec_bits> {}, flat_tpts, flat_td);
            }
        }
    }

    // Accept mutable temporaries
    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE COMMONDX_STL_NAMESPACE::enable_if_t<cute::is_rmem_v<TRC> and
                                                        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TC, CLayout> const& tS, tensor<TRC, CFragLayout>&& tD, Partitioner const& p) {
        copy_fragment<AlignmentInBytes>(tS, tD, p);
    }

    } // namespace detail

    // Public --> gate for element ownership
    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE COMMONDX_STL_NAMESPACE::enable_if_t<cute::is_rmem_v<TRC> and
                                                        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TRC, CFragLayout> const& tS, tensor<TC, CLayout>& tD, Partitioner const& p) {
        if (p.is_thread_active()) {
            detail::copy_fragment<AlignmentInBytes>(tS, tD, p);
        }
    }

    // Accept mutable temporaries
    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE COMMONDX_STL_NAMESPACE::enable_if_t<cute::is_rmem_v<TRC> and
                                                        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TRC, CFragLayout> const& tS, tensor<TC, CLayout>&& tD, Partitioner const& p) {
        copy_fragment<AlignmentInBytes>(tS, tD, p);
    }

    // Public --> gate for element ownership
    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE COMMONDX_STL_NAMESPACE::enable_if_t<cute::is_rmem_v<TRC> and
                                                        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TC, CLayout> const& tS, tensor<TRC, CFragLayout>& tD, Partitioner const& p) {
        if (p.is_thread_active()) {
            detail::copy_fragment<AlignmentInBytes>(tS, tD, p);
        }
    }

    // Accept mutable temporaries
    template<unsigned AlignmentInBytes,
             class TRC,
             class CFragLayout,
             class TC,
             class CLayout,
             class Partitioner>
    CUBLASDX_DEVICE COMMONDX_STL_NAMESPACE::enable_if_t<cute::is_rmem_v<TRC> and
                                                        (cute::is_smem_v<TC> or cute::is_gmem_v<TC>)>
    copy_fragment(tensor<TC, CLayout> const& tS, tensor<TRC, CFragLayout>&& tD, Partitioner const& p) {
        copy_fragment<AlignmentInBytes>(tS, tD, p);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_ACCUMULATOR_COPY_FRAGMENT_HPP
