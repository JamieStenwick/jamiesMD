// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_BARRIERS_MBARRIER_UTCMMA_BARRIER_HPP
#define CUBLASDX_DETAIL_PIPELINE_BARRIERS_MBARRIER_UTCMMA_BARRIER_HPP

#include "cublasdx/detail/pipeline/barriers/barrier_view.hpp"
#include "cublasdx/detail/pipeline/logging.hpp"

namespace cublasdx {
    namespace detail {
        template<int SM, int Threads>
        struct mbarrier_utcmma_barrier {
            static_assert(Threads == 1);

            // Only 1 thread should perform this
            CUBLASDX_DEVICE static void init(barrier_storage_t& ptr) {
                PIPELINE_LOG("mbarrier_utcmma init %d\n", threadIdx.x);
                // Initialize mbarrier
                uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&ptr);
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr), "r"(Threads));
            }

            CUBLASDX_DEVICE
            static void arrive_before(barrier_storage_t&) {}

            // This is a release (signify all smem work done) operation
            // Last thread to complete this is going to switch the shared
            // barrier phase
            CUBLASDX_DEVICE static void arrive_after(barrier_storage_t& ptr) {
                uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&ptr);
                PIPELINE_LOG("mbarrier_utcmma arrive %d %d\n", threadIdx.x, smem_int_ptr);

                // NVVM code movement fence
                asm volatile("" ::: "memory");
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(
                        smem_int_ptr) : "memory");
            };

            // This is a wait operation
            CUBLASDX_DEVICE static bool wait(barrier_storage_t& ptr, uint32_t local_phase) {

                uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&ptr);
                PIPELINE_LOG("mbarrier_utcmma wait %d %d\n", threadIdx.x, smem_int_ptr);

                int32_t wait_complete = 0;

#ifdef __CUDA_ARCH__
                static_assert(__CUDA_ARCH__ >= SM, "Compiling SM<Arch> operator with lower NVCC Architecture");
#endif
                asm volatile("{\n"
                             ".reg .pred                P1;\n"
                             "LAB_WAIT:\n"
                             "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n"
                             "@P1                       bra DONE;\n"
                             "bra                   LAB_WAIT;\n"
                             "DONE:\n"
                             "selp.b32 %0, 1, 0, P1; \n\t"
                             "}\n"
                             : "=r"(wait_complete)
                             : "r"(smem_int_ptr), "r"(local_phase)
                             : "memory");

                return static_cast<bool>(wait_complete);
            }
        };
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_BARRIERS_MBARRIER_UTCMMA_BARRIER_HPP
