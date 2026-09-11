// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp> // cute::logical_divide
#include <cute/swizzle.hpp>         // cute::Swizzle
#include <cute/swizzle_layout.hpp>  // cute::get_nonswizzle_portion
#include <cute/tensor_impl.hpp>     // cute::Tensor
#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_atom.hpp>

#include "cublasdx/detail/cute/extension/algorithm/copy.hpp"

namespace cute {

    struct ld_st_copy {
    };

    struct cp_async_copy {
    };

    // Ugly temporary WAR for getting cooperative_copy copy kind
    // Proper solution: refactor cooperative_copy to get partitioning pattern
    template<uint32_t NumThreads,
             uint32_t MaxVecBits,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout,
             class CopyPolicy = DefaultCopy>
    CUTE_HOST_DEVICE constexpr auto get_cooperative_copy_type(uint32_t const&                     tid,
                                                              Tensor<SrcEngine, SrcLayout> const& src,
                                                              Tensor<DstEngine, DstLayout>        dst,
                                                              CopyPolicy const&                   cpy = {}) {

        auto          common_layout = heuristic_permutation(src, dst);
        Tensor        src_a         = coalesce(logical_divide(src, common_layout), Shape<_1, _1> {});
        Tensor        dst_a         = coalesce(logical_divide(dst, common_layout), Shape<_1, _1> {});
        constexpr int elem_bits     = sizeof_bits_v<typename SrcEngine::value_type>;
        constexpr int total_elem    = size(SrcLayout {});
        constexpr int common_elem   = decltype(max_common_vector(src_a, dst_a))::value;

        if constexpr (total_elem % NumThreads != 0) {
            return ld_st_copy {};
        } else {
            constexpr int total_bits       = total_elem * elem_bits;
            constexpr int max_bits_per_thr = total_bits / NumThreads;
            constexpr int common_bits      = common_elem * elem_bits;
            constexpr int vec_bits = cute::max(elem_bits, cute::gcd(common_bits, int(MaxVecBits), max_bits_per_thr));
            using VecType          = uint_bit_t<vec_bits>;

            constexpr bool is_gmem_to_smem   = is_gmem<SrcEngine>::value && is_smem<DstEngine>::value;
            constexpr bool is_properly_sized = (sizeof_bits_v<VecType> % sizeof_bits_v<float>) == 0;
            constexpr bool is_proper_atom =
                is_same_v<CopyPolicy, cublasdx::detail::auto_copy_async_cache_always_cublasdx> ||
                is_same_v<CopyPolicy, cublasdx::detail::auto_copy_async_cache_global_cublasdx>;

            return conditional_return < is_gmem_to_smem && is_properly_sized &&
                   is_proper_atom > (cp_async_copy {}, ld_st_copy {});
        }

        CUTE_GCC_UNREACHABLE;
    }

} // end namespace cute
