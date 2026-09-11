// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_BARRIERS_MBARRIER_BULK_BARRIER_HPP
#define CUBLASDX_DETAIL_PIPELINE_BARRIERS_MBARRIER_BULK_BARRIER_HPP

#include "cublasdx/detail/pipeline/barriers/barrier_view.hpp"
#include "cublasdx/detail/pipeline/logging.hpp"

namespace cublasdx {
    namespace detail {
        template<int SM, int Threads, int TransactionBytes>
        struct mbarrier_bulk_barrier {
            // Initialize mbarrier to only 1 thread, because only 1 thread needs to launch TMA and
            // expect X bytes from it. If X threads performed expect(N) the number of total bytes
            // would be N * X
            static_assert(Threads == 1);
            static_assert(SM >= 900);
#ifdef __CUDA_ARCH__
            static_assert(__CUDA_ARCH__ >= SM, "Compiling SM<Arch> operator with lower NVCC Architecture");
#endif

            // Only 1 thread should perform this
            CUBLASDX_DEVICE static void init(barrier_storage_t& ptr) {
                PIPELINE_LOG("mbarrier_bulk init %d\n", threadIdx.x);

                uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&ptr);
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr), "r"(Threads) : "memory");
            }

            // This is a release (signify all smem work done) operation
            // Last thread to complete this is going to switch the shared
            // barrier phase
            CUBLASDX_DEVICE static void arrive_before(barrier_storage_t& ptr) {
                PIPELINE_LOG(
                    "mbarrier_bulk arrive and expect %d %d\n", TransactionBytes, cute::cast_smem_ptr_to_uint(&ptr));

                uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&ptr);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_int_ptr),
                             "r"(TransactionBytes)
                             : "memory");
            };

            CUBLASDX_DEVICE static void arrive_after(barrier_storage_t&) {};

            // This is a wait (acquire) operation
            CUBLASDX_DEVICE static void wait(barrier_storage_t& ptr, uint32_t local_phase) {
                PIPELINE_LOG("mbarrier_bulk wait %d\n", cute::cast_smem_ptr_to_uint(&ptr));

                uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&ptr);
                asm volatile("{\n"
                             ".reg .pred                P1;\n"
                             "LAB_WAIT:\n"
                             "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
                             "@P1                       bra DONE;\n"
                             "bra                       LAB_WAIT;\n"
                             "DONE:\n"
                             "}\n"
                             :
                             : "r"(smem_int_ptr), "r"(local_phase)
                             : "memory");
            };
        };
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_BARRIERS_MBARRIER_BULK_BARRIER_HPP
