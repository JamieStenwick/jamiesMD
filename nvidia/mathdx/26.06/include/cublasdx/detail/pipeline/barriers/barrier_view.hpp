// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_BARRIERS_BARRIER_VIEW_HPP
#define CUBLASDX_DETAIL_PIPELINE_BARRIERS_BARRIER_VIEW_HPP

namespace cublasdx {
    namespace detail {
        // Provide an object-like interface for non-owning interactions with barriers and barrier groups
        
        using barrier_storage_t                                  = uint64_t;
        inline constexpr unsigned barrier_buffer_alignment_bytes = alignof(barrier_storage_t);

        template<class Barrier>
        struct barrier_view {
            barrier_storage_t* storage = nullptr;

            CUBLASDX_DEVICE
            barrier_storage_t* data() const { return storage; }

            CUBLASDX_DEVICE
            barrier_storage_t& raw_storage() const { return *storage; }

            CUBLASDX_DEVICE
            uint32_t get_raw_pointer() const { return cute::cast_smem_ptr_to_uint(storage); }

            CUBLASDX_DEVICE
            void init() const { Barrier::init(raw_storage()); }

            CUBLASDX_DEVICE
            void arrive_before() const { Barrier::arrive_before(raw_storage()); }

            CUBLASDX_DEVICE
            void arrive_after() const { Barrier::arrive_after(raw_storage()); }

            template<class Phase>
            CUBLASDX_DEVICE
            auto wait(Phase phase) const { return Barrier::wait(raw_storage(), phase); }

        };

        template<class Barrier>
        CUBLASDX_DEVICE
        barrier_storage_t& raw_barrier_storage(barrier_view<Barrier> barrier) {
            return barrier.raw_storage();
        }

        CUBLASDX_DEVICE
        barrier_storage_t& raw_barrier_storage(barrier_storage_t& barrier) {
            return barrier;
        }

        template<class Barrier, int Count>
        struct barrier_group {
            barrier_storage_t* storage = nullptr;

            CUBLASDX_DEVICE
            void reset(barrier_storage_t* ptr) { storage = ptr; }

            CUBLASDX_DEVICE
            barrier_storage_t* data() const { return storage; }

            CUBLASDX_DEVICE
            constexpr int size() const { return Count; }

            CUBLASDX_DEVICE
            barrier_view<Barrier> operator[](int idx) const { return {storage + idx}; }

            CUBLASDX_DEVICE
            void init_all() const {
                CUTE_UNROLL
                for (int i = 0; i < Count; ++i) {
                    (*this)[i].init();
                }
            }
        };
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_BARRIERS_BARRIER_VIEW_HPP
