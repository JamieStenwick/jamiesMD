// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_SHARED_MEMORY_HPP
#define CUBLASDX_DETAIL_SHARED_MEMORY_HPP

#ifndef COMMONDX_DEVICE_SLICE
#define COMMONDX_DEVICE_SLICE __host__ __device__ __forceinline__
#endif

#include "commondx/shared_memory.hpp"

#include "cublasdx/detail/blas_execution/blas_execution.hpp"
#include "cublasdx/detail/tensor.hpp"

namespace cublasdx {
    namespace shared_memory {
        using commondx::shared_memory::slice;
        using commondx::shared_memory::slice_into_pointers;
        using commondx::shared_memory::slice_into_tensors;
    } // namespace shared_memory

    using commondx::make_shared_storage_calculator;

    template<class BLAS,
             class AValueType = typename BLAS::a_value_type,
             class BValueType = typename BLAS::b_value_type,
             class CValueType = typename BLAS::c_value_type,
             class ALayout,
             class BLayout,
             class CLayout,
             __CUTE_REQUIRES(
                 cublasdx::is_layout_v<ALayout>and cublasdx::is_layout_v<BLayout>and cublasdx::is_layout_v<CLayout>)>
    CUBLASDX_HOST_DEVICE constexpr COMMONDX_STL_NAMESPACE::enable_if_t<is_blas_execution<BLAS>::value, unsigned>
    get_shared_storage_size(ALayout const& a_layout, BLayout const& b_layout, CLayout const& c_layout) {
        unsigned requirement = 0;
        requirement          = commondx::detail::add_aligned_shared_memory_extent(
            requirement, alignment_of<BLAS>::a, sizeof(AValueType), a_layout);
        requirement = commondx::detail::add_aligned_shared_memory_extent(
            requirement, alignment_of<BLAS>::b, sizeof(BValueType), b_layout);
        requirement = commondx::detail::add_aligned_shared_memory_extent(
            requirement, alignment_of<BLAS>::c, sizeof(CValueType), c_layout);
        return requirement;
    }

    template<class BLAS,
             class AValueType = typename BLAS::a_value_type,
             class BValueType = typename BLAS::b_value_type,
             class CValueType = typename BLAS::c_value_type>
    CUBLASDX_HOST_DEVICE constexpr COMMONDX_STL_NAMESPACE::enable_if_t<is_blas_execution<BLAS>::value, unsigned>
    get_shared_storage_size(unsigned lda = BLAS::lda,
                            unsigned ldb = BLAS::ldb,
                            unsigned ldc = BLAS::ldc) {
        return get_shared_storage_size<BLAS, AValueType, BValueType, CValueType>(
            BLAS::get_layout_smem_a(lda), BLAS::get_layout_smem_b(ldb), BLAS::get_layout_smem_c(ldc));
    }

    template<class BLAS,
             class AValueType = typename BLAS::a_value_type,
             class BValueType = typename BLAS::b_value_type,
             class ALayout,
             class BLayout,
             __CUTE_REQUIRES(cublasdx::is_layout_v<ALayout>and cublasdx::is_layout_v<BLayout>)>
    CUBLASDX_HOST_DEVICE constexpr COMMONDX_STL_NAMESPACE::enable_if_t<is_blas_execution<BLAS>::value, unsigned>
    get_shared_storage_size_ab(ALayout const& a_layout, BLayout const& b_layout) {
        unsigned requirement = 0;
        requirement          = commondx::detail::add_aligned_shared_memory_extent(
            requirement, alignment_of<BLAS>::a, sizeof(AValueType), a_layout);
        requirement = commondx::detail::add_aligned_shared_memory_extent(
            requirement, alignment_of<BLAS>::b, sizeof(BValueType), b_layout);
        return requirement;
    }

    template<class BLAS, class AValueType = typename BLAS::a_value_type, class BValueType = typename BLAS::b_value_type>
    CUBLASDX_HOST_DEVICE constexpr COMMONDX_STL_NAMESPACE::enable_if_t<is_blas_execution<BLAS>::value, unsigned>
    get_shared_storage_size_ab(const unsigned lda = BLAS::lda,
                               const unsigned ldb = BLAS::ldb) {
        return get_shared_storage_size_ab<BLAS, AValueType, BValueType>(BLAS::get_layout_smem_a(lda),
                                                                        BLAS::get_layout_smem_b(ldb));
    }

    template<class BLAS,
             class AValueType = typename BLAS::a_value_type,
             class BValueType = typename BLAS::b_value_type,
             class CValueType = typename BLAS::c_value_type,
             class ALayout    = decltype(BLAS::get_layout_smem_a().layout),
             class BLayout    = decltype(BLAS::get_layout_smem_b().layout),
             class CLayout    = decltype(BLAS::get_layout_smem_c().layout),
             __CUTE_REQUIRES(
                 cublasdx::is_layout_v<ALayout>and cublasdx::is_layout_v<BLayout>and cublasdx::is_layout_v<CLayout>)>
    CUBLASDX_DEVICE auto slice_shared_memory(void*          smem,
                                             ALayout const& a_layout = {},
                                             BLayout const& b_layout = {},
                                             CLayout const& c_layout = {}) {
        static_assert(is_complete_blas<BLAS>::value, "Can't slice shared memory, description is not complete");
        static_assert(function_of<BLAS>::value == function::MM, "The 3 argument version of slice_shared_memory can only be used with GEMM");
        
        // Call slice_into_tensors with (alignment, layout) pairs:
        return shared_memory::slice_into_pointers<AValueType, BValueType, CValueType>(smem,
                                                                                      alignment_of_v_a<BLAS>,
                                                                                      cublasdx::cosize(a_layout),
                                                                                      alignment_of_v_b<BLAS>,
                                                                                      cublasdx::cosize(b_layout),
                                                                                      alignment_of_v_c<BLAS>,
                                                                                      cublasdx::cosize(c_layout));
    }

    template<class BLAS,
             class AValueType = typename BLAS::a_value_type,
             class BValueType = typename BLAS::b_value_type,
             class CValueType = typename BLAS::c_value_type>
    CUBLASDX_DEVICE auto slice_shared_memory(void* smem, unsigned lda, unsigned ldb, unsigned ldc) {
        static_assert(function_of<BLAS>::value == function::MM, "The 3 argument version of slice_shared_memory can only be used with GEMM");
        return shared_memory::slice_into_pointers<AValueType, BValueType, CValueType>(
            smem,
            alignment_of_v_a<BLAS>,
            cublasdx::cosize(BLAS::get_layout_smem_a(lda)),
            alignment_of_v_b<BLAS>,
            cublasdx::cosize(BLAS::get_layout_smem_b(ldb)),
            alignment_of_v_c<BLAS>,
            cublasdx::cosize(BLAS::get_layout_smem_c(ldc)));
    }

    template<class BLAS,
             class AValueType = typename BLAS::a_value_type,
             class BValueType = typename BLAS::b_value_type,
             class ALayout    = decltype(BLAS::get_layout_smem_a().layout),
             class BLayout    = decltype(BLAS::get_layout_smem_b().layout),
             __CUTE_REQUIRES(cublasdx::is_layout_v<ALayout>and cublasdx::is_layout_v<BLayout>)>
    CUBLASDX_DEVICE auto slice_shared_memory_ab(void*          smem,
                                                ALayout const& a_layout = {},
                                                BLayout const& b_layout = {}) {
        static_assert(is_complete_blas<BLAS>::value, "Can't slice shared memory, description is not complete");

        // Switch to slice_into_tensors with (alignment, layout) pairs:
        return shared_memory::slice_into_pointers<AValueType, BValueType>(smem,
                                                                          alignment_of_v_a<BLAS>,
                                                                          cublasdx::cosize(a_layout),
                                                                          alignment_of_v_b<BLAS>,
                                                                          cublasdx::cosize(b_layout));
    }

    template<class BLAS, class AValueType = typename BLAS::a_value_type, class BValueType = typename BLAS::b_value_type>
    CUBLASDX_DEVICE auto slice_shared_memory_ab(void* smem, const unsigned lda, const unsigned ldb) {
        return shared_memory::slice_into_pointers<AValueType, BValueType>(
            smem,
            alignment_of_v_a<BLAS>,
            cublasdx::cosize(BLAS::get_layout_smem_a(lda)),
            alignment_of_v_b<BLAS>,
            cublasdx::cosize(BLAS::get_layout_smem_b(ldb)));
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_SHARED_MEMORY_HPP
