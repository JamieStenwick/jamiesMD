// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_BARRIERS_BARRIER_HPP
#define CUBLASDX_DETAIL_PIPELINE_BARRIERS_BARRIER_HPP

#include "cublasdx/detail/copy.hpp"

#include "cublasdx/detail/pipeline/barriers/atomic_sync_barrier.hpp"
#include "cublasdx/detail/pipeline/barriers/mbarrier_sync_barrier.hpp"
#include "cublasdx/detail/pipeline/barriers/mbarrier_utcmma_barrier.hpp"
#include "cublasdx/detail/pipeline/barriers/mbarrier_async_barrier.hpp"
#include "cublasdx/detail/pipeline/barriers/mbarrier_bulk_barrier.hpp"

namespace cublasdx {
    namespace detail {
        template<copy_kind CopyKind, int SM, int Threads, int BarrierBytes>
        CUBLASDX_HOST_DEVICE constexpr auto get_copy_smem_barrier() {
            constexpr bool is_atomic_sync    = (CopyKind == copy_kind::sync) and (SM < 800);
            constexpr bool is_mbarrier_sync  = (CopyKind == copy_kind::sync) and (SM >= 800);
            constexpr bool is_mbarrier_async = (CopyKind == copy_kind::async) and (SM >= 800);
            constexpr bool is_mbarrier_bulk  = (CopyKind == copy_kind::bulk) and (SM >= 900);

            if constexpr (is_atomic_sync) {
                return detail::atomic_sync_barrier<SM, Threads> {};
            } else if constexpr (is_mbarrier_sync) {
                return detail::mbarrier_sync_barrier<SM, Threads> {};
            } else if constexpr (is_mbarrier_async) {
                return detail::mbarrier_async_barrier<SM, Threads> {};
            } else if constexpr (is_mbarrier_bulk) {
                return detail::mbarrier_bulk_barrier<SM, Threads, BarrierBytes> {};
            } else {
                static_assert(is_atomic_sync or is_mbarrier_sync or is_mbarrier_async or is_mbarrier_bulk,
                              "Unsupported barrier requirements");
            }

            CUTE_GCC_UNREACHABLE;
        }

        template<instruction_type InstructionKind, int SM, int Threads>
        CUBLASDX_HOST_DEVICE constexpr auto get_compute_smem_barrier() {
            constexpr bool is_atomic_sync   = (InstructionKind != instruction_type::utcmma) and (SM < 800);
            constexpr bool is_mbarrier_sync = (InstructionKind != instruction_type::utcmma) and (SM >= 800);
            constexpr bool is_utcmma_sync   = InstructionKind == instruction_type::utcmma;

            static_assert(not is_utcmma_sync or (SM == 1000 or SM == 1030 or SM == 1100));

            if constexpr (is_atomic_sync) {
                return detail::atomic_sync_barrier<SM, Threads> {};
            } else if constexpr (is_utcmma_sync) {
                return detail::mbarrier_utcmma_barrier<SM, Threads> {};
            } else if constexpr (is_mbarrier_sync) {
                return detail::mbarrier_sync_barrier<SM, Threads> {};
            } else {
                static_assert(is_atomic_sync or is_mbarrier_sync or is_utcmma_sync, "Unsupported barrier requirements");
            }

            CUTE_GCC_UNREACHABLE;
        }
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_BARRIERS_BARRIER_HPP
