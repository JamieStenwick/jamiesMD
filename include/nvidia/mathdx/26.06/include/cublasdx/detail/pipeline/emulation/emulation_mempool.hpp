// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_MEMPOOL_HPP
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_MEMPOOL_HPP

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_MEMPOOL_MAX_DEVICES
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_MEMPOOL_MAX_DEVICES 64
#endif

namespace cublasdx {
    namespace detail {
        class emulation_mempool_cache {
            static constexpr int max_devices = CUBLASDX_DETAIL_PIPELINE_EMULATION_MEMPOOL_MAX_DEVICES;
            cudaMemPool_t pools_[max_devices] = {};
            std::mutex mutex_;

        public:
            ~emulation_mempool_cache() {
                for (auto pool : pools_) {
                    if (pool != nullptr) {
                        cudaMemPoolDestroy(pool);
                    }
                }
            }

            cudaMemPool_t get(int const device) {
                if (device < 0 || device >= max_devices) {
                    return nullptr;
                }

                const std::lock_guard<std::mutex> lock(mutex_);
                if (pools_[device] != nullptr) {
                    return pools_[device];
                }

                cudaMemPoolProps pool_props {};
                pool_props.allocType     = cudaMemAllocationTypePinned;
                pool_props.location.type = cudaMemLocationTypeDevice;
                pool_props.location.id   = device;

                cudaMemPool_t pool = nullptr;
                cudaError_t err = cudaMemPoolCreate(&pool, &pool_props);
                if (err != cudaSuccess) {
                    cudaGetLastError();
                    return nullptr;
                }

                std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
                err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
                if (err != cudaSuccess) {
                    cudaGetLastError();
                    cudaMemPoolDestroy(pool);
                    return nullptr;
                }

                pools_[device] = pool;
                return pool;
            }
        };

        inline cudaMemPool_t get_emulation_mempool(int const device) {
            static emulation_mempool_cache cache;
            return cache.get(device);
        }

        inline cudaError_t emulation_malloc_async(void** ptr, std::size_t const bytes, cudaStream_t const stream) {
            int device = 0;
            cudaError_t err = cudaGetDevice(&device);
            if (err != cudaSuccess) {
                return err;
            }

            cudaMemPool_t pool = get_emulation_mempool(device);
            if (pool == nullptr) {
                return cudaMallocAsync(ptr, bytes, stream);
            }
            return cudaMallocFromPoolAsync(ptr, bytes, pool, stream);
        }
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_EMULATION_MEMPOOL_HPP
