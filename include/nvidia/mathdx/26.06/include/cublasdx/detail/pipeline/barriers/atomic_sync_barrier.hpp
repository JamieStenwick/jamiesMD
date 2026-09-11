// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_BARRIERS_ATOMIC_SYNC_BARRIER_HPP
#define CUBLASDX_DETAIL_PIPELINE_BARRIERS_ATOMIC_SYNC_BARRIER_HPP

#include "cublasdx/detail/pipeline/barriers/barrier_view.hpp"
#include "cublasdx/detail/pipeline/logging.hpp"

namespace cublasdx {
    namespace detail {
        template<int SM, int Threads>
        struct atomic_sync_barrier {
            // max_threads must be a power of 2 for bit-masking
            static constexpr uint32_t max_threads = 1024;
            static_assert(Threads <= max_threads);

            static constexpr uint32_t barrier_bits     = cute::sizeof_bits_v<barrier_storage_t>;
            static constexpr uint32_t phase_shift      = barrier_bits - 1;
            static constexpr uint32_t threads_bit_mask = max_threads + (max_threads - 1);

            // Most significant bit is the current shared-phase of the barrier
            // Set it to 1 on start (local starting phase is 0 for each thread)
            static constexpr barrier_storage_t init_val = 1ull << phase_shift;

            // Only 1 thread should perform this
            CUBLASDX_DEVICE static void init(barrier_storage_t& ptr) {
                PIPELINE_LOG("atomic_barrier init %d\n", threadIdx.x);
                ptr = init_val;
            } // with syncthreads afterwards

            CUBLASDX_DEVICE static void arrive_before(barrier_storage_t& ptr) {};

            // This is a release (signify all smem work done) operation
            // Last thread to complete this is going to switch the shared
            // barrier phase
            CUBLASDX_DEVICE static void arrive_after(barrier_storage_t& ptr) {
                PIPELINE_LOG("atomic_barrier arrive %d\n", threadIdx.x);

                barrier_storage_t value {};
                uint32_t          smem_int_ptr = cute::cast_smem_ptr_to_uint(&ptr);

                // Mark this thread as done (increment "waiting" counter)
                asm volatile("atom.release.cta.shared.add.u64 %0, [%1], 1;"
                             : "=l"(value)
                             : "r"(smem_int_ptr)
                             : "memory");

                value += 1;

                // If this is the last thread, flip the shared phase bit
                if ((value & threads_bit_mask) == Threads) {
                    PIPELINE_LOG("atomic_barrier flip %d\n", threadIdx.x);
                    const barrier_storage_t flipped_empty_state = (value & init_val) ^ init_val;
                    asm volatile("st.relaxed.cta.shared.b64 [%0], %1;"
                                 :
                                 : "r"(smem_int_ptr), "l"(flipped_empty_state)
                                 : "memory");
                }
            };

            // This is a wait operation
            CUBLASDX_DEVICE static void wait(barrier_storage_t& ptr, barrier_storage_t local_phase) {
                PIPELINE_LOG("atomic_barrier wait %d\n", threadIdx.x);
                uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&ptr);
                asm volatile("{\n\t"
                             ".reg .u64 val;\n\t"
                             ".reg .pred P1;\n\t"
                             "LAB_WAIT:\n\t"
                             "ld.acquire.cta.shared.u64 val, [%0];\n\t"
                             "shr.u64 val, val, %1;\n\t"
                             "setp.eq.u64 P1, val, %2;\n\t"
                             "@P1 bra DONE;\n\t"
                             "bra LAB_WAIT;\n\t"
                             "DONE:\n\t"
                             "}\n\t"
                             :
                             : "r"(smem_int_ptr), "r"(phase_shift), "l"(local_phase)
                             : "memory");
            };
        };
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_BARRIERS_ATOMIC_SYNC_BARRIER_HPP
