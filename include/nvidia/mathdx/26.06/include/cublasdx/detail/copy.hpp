// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_COPY_HPP
#define CUBLASDX_DETAIL_COPY_HPP

#include "cublasdx/detail/blas_backend.hpp"

namespace cublasdx {

    namespace detail {
        enum class copy_kind
        {
            sync,
            async,
            bulk
        };

        template<class T>
        struct convert_cute_to_copy_kind;

        template<>
        struct convert_cute_to_copy_kind<cute::ld_st_copy>: cute::integral_constant<copy_kind, copy_kind::sync> {
        };

        template<>
        struct convert_cute_to_copy_kind<cute::cp_async_copy>: cute::integral_constant<copy_kind, copy_kind::async> {
        };

        template<class T>
        inline constexpr copy_kind convert_cute_to_copy_kind_v = convert_cute_to_copy_kind<T>::value;

        template<class... CopyKinds>
        CUBLASDX_HOST_DEVICE constexpr copy_kind choose_most_conservative_copy_instruction(CopyKinds... args) {
            const bool is_any_sync  = ((args == copy_kind::sync) || ...);
            const bool is_any_async = ((args == copy_kind::async) || ...);
            const bool are_all_bulk = ((args == copy_kind::bulk) && ...);

            if (is_any_sync) {
                return copy_kind::sync;
            } else if (is_any_async) {
                return copy_kind::async;
            } else if (are_all_bulk) {
                return copy_kind::bulk;
            }

            // Unexpected safe exit
            assert(is_any_sync or is_any_async or are_all_bulk);
            return copy_kind::sync;
        }

        inline constexpr unsigned tma_buffer_alignment_bytes = 128;

        template<class ElemType, class Layout>
        CUBLASDX_HOST_DEVICE constexpr auto is_tma_compatible_layout(Layout layout) {
            constexpr int  size_1        = cute::size<0>(layout);
            constexpr int  size_2        = cute::size<1>(layout);
            constexpr bool less_than_256 = (size_1 <= 256) and (size_2 <= 256);

            if constexpr (not cute::is_static_v<Layout> or not less_than_256) {
                return cute::false_type {};
            } else {
                auto tma_compatible_atoms = cute::make_tuple(cute::GMMA::Layout_MN_SW128_Atom<ElemType> {},
                                                             cute::GMMA::Layout_MN_SW64_Atom<ElemType> {},
                                                             cute::GMMA::Layout_MN_SW32_Atom<ElemType> {},
                                                             cute::GMMA::Layout_MN_INTER_Atom<ElemType> {},
                                                             cute::GMMA::Layout_K_SW128_Atom<ElemType> {},
                                                             cute::GMMA::Layout_K_SW64_Atom<ElemType> {},
                                                             cute::GMMA::Layout_K_SW32_Atom<ElemType> {},
                                                             cute::GMMA::Layout_K_INTER_Atom<ElemType> {});

                auto is_compatible = [=](auto value, auto elem) constexpr {
                    if constexpr (cute::evenly_divides(cute::shape(Layout {}), cute::shape(decltype(elem) {}))) {
                        auto           tiled_col = remove_zeros_from_layout(cute::tile_to_shape(elem, cute::shape(layout), cute::GenColMajor {}));
                        auto           tiled_row = remove_zeros_from_layout(cute::tile_to_shape(elem, cute::shape(layout), cute::GenRowMajor {}));
                        constexpr bool compat =
                            cute::is_same_v<cute::remove_cvref_t<decltype(tiled_col)>, cute::remove_cvref_t<Layout>> or
                            cute::is_same_v<cute::remove_cvref_t<decltype(tiled_row)>, cute::remove_cvref_t<Layout>>;
                        return value || cute::C<compat> {};
                    } else {
                        return value || cute::false_type {};
                    }
                };

                constexpr bool result =
                    cute::fold(decltype(tma_compatible_atoms) {}, cute::false_type {}, is_compatible);

                return cute::C<result> {};
            }

            CUTE_GCC_UNREACHABLE;
        }
    } // namespace detail

    CUBLASDX_DEVICE
    void copy_wait() {
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();
    }

    template<uint32_t NumThreads, class SrcLayout, class DstLayout>
    CUBLASDX_HOST_DEVICE constexpr auto force_naive_copy(SrcLayout, DstLayout) {
        constexpr int max_vec_src = decltype(cute::max_alignment(SrcLayout {}))::value;
        constexpr int max_vec_dst = decltype(cute::max_alignment(DstLayout {}))::value;
        constexpr int gcd_max_vec = cute::gcd(max_vec_src, max_vec_dst);
        
        return cute::constant<bool, gcd_max_vec % 2 != 0> {};
    }

    template<uint32_t NumThreads, uint32_t AlignmentInBytes, uint32_t SM, class SrcTensor, class DstTensor>
    CUBLASDX_HOST_DEVICE constexpr auto get_copy_type(SrcTensor, DstTensor) {
        using src_shape = decltype(cute::shape(SrcTensor {}));
        using dst_shape = decltype(cute::shape(DstTensor {}));
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
                      "cublasdx::copy requires static tensor layouts");
            
        constexpr bool force_naive = force_naive_copy<NumThreads>(cute::layout(SrcTensor {}), cute::layout(DstTensor {}));

        if constexpr (force_naive) {
            return cute::ld_st_copy {};
        } else {
            using src_elem_with_const      = cute::remove_reference_t<typename SrcTensor::reference>;
            constexpr bool is_const_source = cute::is_const_v<src_elem_with_const>;
            constexpr auto copy_policy_async =
                cute::conditional_return<is_const_source>(cublasdx::detail::auto_copy_async_cache_global_cublasdx {},
                                                          cublasdx::detail::auto_copy_async_cache_always_cublasdx {});
            constexpr auto copy_policy_sync = cute::DefaultCopy {};
    
            constexpr auto copy_policy = cute::conditional_return<(SM < 800)>(copy_policy_sync, copy_policy_async);
    
            constexpr int max_vec_bits = cute::gcd(
                AlignmentInBytes * 8,
                cute::max_alignment(cute::layout(SrcTensor {})) * cute::sizeof_bits_v<typename SrcTensor::value_type>,
                cute::max_alignment(cute::layout(DstTensor {})) * cute::sizeof_bits_v<typename DstTensor::value_type>);
    
            return decltype(cute::get_cooperative_copy_type<NumThreads, max_vec_bits>(
                0, SrcTensor {}, DstTensor {}, copy_policy)) {};
        }

        CUTE_GCC_UNREACHABLE;
    }

    namespace detail {
        template<uint32_t NumThreads,
                 uint32_t AlignmentInBytes,
                 class SrcEngine,
                 class SrcLayout,
                 class DstEngine,
                 class DstLayout>
        CUBLASDX_DEVICE void copy_sync_impl(const unsigned int                            tid,
                                            const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                                            cublasdx::tensor<DstEngine, DstLayout>&       dst) {
            using src_shape = decltype(src.shape());
            using dst_shape = decltype(dst.shape());
            static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
                          "cublasdx::copy requires static tensor layouts");

            constexpr bool force_naive = force_naive_copy<NumThreads>(SrcLayout {}, DstLayout {});

            if constexpr (force_naive) {
                cute::naive_cooperative_copy<NumThreads>(tid, src, dst);
            } else {
                constexpr int max_vec_bits = cute::gcd(
                    AlignmentInBytes * 8,
                    cute::max_alignment(SrcLayout {}) * cute::sizeof_bits_v<typename SrcEngine::value_type>,
                    cute::max_alignment(DstLayout {}) * cute::sizeof_bits_v<typename DstEngine::value_type>);

                // Coalesce to simplify the layout for the copy
                auto tmp_src = cute::make_tensor(src.data(), cute::coalesce(src.layout()));
                auto tmp_dst = cute::make_tensor(dst.data(), cute::coalesce(dst.layout()));

                cute::cooperative_copy<NumThreads, max_vec_bits>(tid, tmp_src, tmp_dst);
            }
        }

        template<uint32_t NumThreads,
                 uint32_t AlignmentInBytes,
                 class SrcEngine,
                 class SrcLayout,
                 class DstEngine,
                 class DstLayout>
        CUBLASDX_DEVICE void copy_impl(const unsigned int                            tid,
                                       const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                                       cublasdx::tensor<DstEngine, DstLayout>&       dst) {
            using src_shape = decltype(src.shape());
            using dst_shape = decltype(dst.shape());
            static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
                          "cublasdx::copy requires static tensor layouts");

            constexpr bool force_naive = force_naive_copy<NumThreads>(SrcLayout {}, DstLayout {});

            if constexpr (force_naive) {
                cute::naive_cooperative_copy<NumThreads>(tid, src, dst);
            } else {
                using src_elem_with_const = cute::remove_reference_t<typename SrcEngine::reference>;

                constexpr bool is_const_source = cute::is_const_v<src_elem_with_const>;
                constexpr auto copy_policy =
                    cute::conditional_return<is_const_source>(cublasdx::detail::auto_copy_async_cache_global_cublasdx {},
                                                              cublasdx::detail::auto_copy_async_cache_always_cublasdx {});

                constexpr int max_vec_bits =
                cute::gcd(AlignmentInBytes * 8,
                          cute::max_alignment(SrcLayout {}) * cute::sizeof_bits_v<typename SrcEngine::value_type>,
                          cute::max_alignment(DstLayout {}) * cute::sizeof_bits_v<typename DstEngine::value_type>);

                auto tmp_src = cute::make_tensor(src.data(), cute::coalesce(src.layout()));
                auto tmp_dst = cute::make_tensor(dst.data(), cute::coalesce(dst.layout()));
                cute::cooperative_copy<NumThreads, max_vec_bits>(tid, tmp_src, tmp_dst, copy_policy);
            }
        }
    } // namespace detail

    template<uint32_t NumThreads,
             uint32_t AlignmentInBytes,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout>
    CUBLASDX_DEVICE void copy_sync(const unsigned int                            tid,
                                   const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                                   cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        if (tid < NumThreads) {
            detail::copy_sync_impl<NumThreads, AlignmentInBytes>(tid, src, dst);
        }
    }

    // Allow mutable temporaries
    template<uint32_t NumThreads,
             uint32_t AlignmentInBytes,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout>
    CUBLASDX_DEVICE auto copy_sync(const unsigned int                            tid,
                                   const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                                   cublasdx::tensor<DstEngine, DstLayout>&&      dst) {
        return copy_sync<NumThreads, AlignmentInBytes>(tid, src, dst);
    }

    template<uint32_t NumThreads,
             uint32_t AlignmentInBytes,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout>
    CUBLASDX_DEVICE void copy(const unsigned int                            tid,
                              const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        if (tid < NumThreads) {
            detail::copy_impl<NumThreads, AlignmentInBytes>(tid, src, dst);
        }
    }

    // Allow mutable temporaries
    template<uint32_t NumThreads,
             uint32_t AlignmentInBytes,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout>
    CUBLASDX_DEVICE auto copy(const unsigned int                            tid,
                              const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                              cublasdx::tensor<DstEngine, DstLayout>&&      dst) {
        return copy<NumThreads, AlignmentInBytes>(tid, src, dst);
    }

    template<uint32_t NumThreads, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUBLASDX_DEVICE auto copy(const unsigned int                            tid,
                              const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        constexpr unsigned int max_vec_bytes = sizeof(typename SrcEngine::value_type);
        return copy<NumThreads, max_vec_bytes>(tid, src, dst);
    }

    // This overload uses as many threads as defined in BLAS::block_dim
    template<class BLAS, uint32_t AlignmentInBytes, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUBLASDX_DEVICE auto copy(const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        using src_shape = decltype(src.shape());
        using dst_shape = decltype(dst.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
                      "cublasdx::copy requires static tensor layouts");
        constexpr unsigned int num_threads    = BLAS::block_dim.x * BLAS::block_dim.y * BLAS::block_dim.z;
        constexpr unsigned int block_dim_rank = (BLAS::block_dim.z > 1) ? 3 : ((BLAS::block_dim.y > 1) ? 2 : 1);
        unsigned int           tid            = detail::get_thread_idx<block_dim_rank>();
        return copy<num_threads, AlignmentInBytes>(tid, src, dst);
    }

    // Allow mutable temporaries
    template<class BLAS, uint32_t AlignmentInBytes, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUBLASDX_DEVICE auto copy(const cublasdx::tensor<SrcEngine, SrcLayout>& src,
                              cublasdx::tensor<DstEngine, DstLayout>&&      dst) {
        return copy<BLAS, AlignmentInBytes>(src, dst);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_COPY_HPP
