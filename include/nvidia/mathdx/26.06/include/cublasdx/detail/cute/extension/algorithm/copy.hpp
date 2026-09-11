#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ALGORITHM_COPY_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ALGORITHM_COPY_HPP

#include <cute/algorithm/copy.hpp>

#include "cublasdx/detail/cute/extension/arch/copy.hpp"

namespace cute {

    //
    // copy_if -- AutoCopyAsyncCacheAlways
    //
    template<class PrdTensor, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUTE_HOST_DEVICE void copy_if(cublasdx::detail::auto_copy_async_cache_always_cublasdx const& /*cpy*/,
                                  PrdTensor const&                    pred,
                                  Tensor<SrcEngine, SrcLayout> const& src,
                                  Tensor<DstEngine, DstLayout>&       dst) {
        using SrcType = typename SrcEngine::value_type;
        using DstType = typename DstEngine::value_type;

        auto copy_op = []() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
            if constexpr (is_gmem<SrcEngine>::value && is_smem<DstEngine>::value &&
                          sizeof(SrcType) == sizeof(DstType) &&
                          (sizeof(SrcType) == 4 || sizeof(SrcType) == 8 || sizeof(SrcType) == 16)) {
                return SM80_CP_ASYNC_CACHEALWAYS<SrcType, DstType> {};
            } else {
                return UniversalCopy<SrcType, DstType> {};
            }

            CUTE_GCC_UNREACHABLE;
#else
            return UniversalCopy<SrcType, DstType> {};
#endif
        }();

        CUTE_UNROLL
        for (int i = 0; i < size(dst); ++i) {
            if (pred(i)) {
                copy_op.copy(src(i), dst(i));
            }
        }
    }

    //
    // copy -- AutoCopyAsyncCacheAlways
    //

    template<class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUTE_HOST_DEVICE void copy(cublasdx::detail::auto_copy_async_cache_always_cublasdx const& cpy,
                               Tensor<SrcEngine, SrcLayout> const&                            src, // (V,Rest...)
                               Tensor<DstEngine, DstLayout>&                                  dst)                                  // (V,Rest...)
    {
        copy_if(cpy, constant_fn<true_type> {}, src, dst);
    }

    //
    // copy_if -- AutoCopyAsyncCacheGlobal
    //
    template<class PrdTensor, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUTE_HOST_DEVICE void copy_if(cublasdx::detail::auto_copy_async_cache_global_cublasdx const& /*cpy*/,
                                  PrdTensor const&                    pred,
                                  Tensor<SrcEngine, SrcLayout> const& src,
                                  Tensor<DstEngine, DstLayout>&       dst) {
        using SrcType = typename SrcEngine::value_type;
        using DstType = typename DstEngine::value_type;

        auto copy_op = []() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
            if constexpr (is_gmem<SrcEngine>::value && is_smem<DstEngine>::value &&
                          sizeof(SrcType) == sizeof(DstType)) {
                if constexpr (sizeof(SrcType) == 16) {
                    return SM80_CP_ASYNC_CACHEGLOBAL<SrcType, DstType> {};
                } else if constexpr (sizeof(SrcType) == 4 || sizeof(SrcType) == 8) {
                    return SM80_CP_ASYNC_CACHEALWAYS<SrcType, DstType> {};
                } else {
                    return UniversalCopy<SrcType, DstType> {};
                }
            } else {
                return UniversalCopy<SrcType, DstType> {};
            }

            CUTE_GCC_UNREACHABLE;
#else
            return UniversalCopy<SrcType, DstType> {};
#endif
        }();

        CUTE_UNROLL
        for (int i = 0; i < size(dst); ++i) {
            if (pred(i)) {
                copy_op.copy(src(i), dst(i));
            }
        }
    }

    //
    // copy -- AutoCopyAsyncCacheGlobal
    //

    template<class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    CUTE_HOST_DEVICE void copy(cublasdx::detail::auto_copy_async_cache_global_cublasdx const& cpy,
                               Tensor<SrcEngine, SrcLayout> const&                            src, // (V,Rest...)
                               Tensor<DstEngine, DstLayout>&                                  dst)                                  // (V,Rest...)
    {
        copy_if(cpy, constant_fn<true_type> {}, src, dst);
    }

} // namespace cute

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ALGORITHM_COPY_HPP
