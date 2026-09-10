// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_RESULT_HPP
#define CUBLASDX_DETAIL_PIPELINE_RESULT_HPP

#if !defined(__CUDACC_RTC__)
#    include <cuda_runtime.h>
#endif

#include "cublasdx/detail/expected.hpp"
#include "cublasdx/types.hpp"

namespace cublasdx {
    enum class pipeline_error_code
    {
        none,
        incompatible_k_dimensions,
        tile_divisibility,
        insufficient_stages,
        emulation_k_dimension_too_large,
        allocation_failed,
        unknown,
        cuda
    };

    class pipeline_error {
        int cuda_error_ = 0;

        static constexpr pipeline_error_code normalize_code(pipeline_error_code const code, int const cuda_error) {
            return (code == pipeline_error_code::none && cuda_error != 0) ? pipeline_error_code::cuda : code;
        }

    public:
        pipeline_error_code code = pipeline_error_code::none;

        constexpr pipeline_error() = default;

        constexpr pipeline_error(pipeline_error_code const code, int const cuda_error = 0):
            cuda_error_(cuda_error),
            code(normalize_code(code, cuda_error)) {}

        explicit constexpr pipeline_error(int const cuda_error):
            cuda_error_(cuda_error),
            code(normalize_code(pipeline_error_code::none, cuda_error)) {}

#if defined(__CUDACC_RTC__)
        CUBLASDX_DEVICE int get_cuda_error() const {
            return cuda_error_;
        }
#else
        cudaError_t get_cuda_error() const {
            return static_cast<cudaError_t>(cuda_error_);
        }
#endif
    };

    inline constexpr char const* pipeline_error_string(pipeline_error_code const code) {
        switch (code) {
            case pipeline_error_code::none:
                return "none";
            case pipeline_error_code::incompatible_k_dimensions:
                return "incompatible K dimensions";
            case pipeline_error_code::tile_divisibility:
                return "global dimensions are not divisible by the BLAS tile";
            case pipeline_error_code::insufficient_stages:
                return "problem K dimension does not provide enough pipeline stages";
            case pipeline_error_code::emulation_k_dimension_too_large:
                return "emulation K dimension exceeds 33025; "
                       "int8-sliced emulation accumulates 8-bit products into int32 internally, "
                       "use split-K for larger emulation pipelines";
            case pipeline_error_code::allocation_failed:
                return "temporary emulation allocation failed";
            case pipeline_error_code::unknown:
                return "unknown error";
            case pipeline_error_code::cuda:
                return "CUDA runtime error";
        }
        return "unknown error";
    }

    namespace detail {
        template<class DevicePipeline>
        class host_pipeline {
            DevicePipeline device_pipeline_;

        public:
            using device_pipeline_type = DevicePipeline;
            static constexpr int max_threads_per_block = DevicePipeline::max_threads_per_block;

            explicit host_pipeline(DevicePipeline const& device_pipeline):
                device_pipeline_(device_pipeline) {}

            explicit host_pipeline(DevicePipeline&& device_pipeline):
                device_pipeline_(static_cast<DevicePipeline&&>(device_pipeline)) {}

            device_pipeline_type get_device_handle() const& {
                return device_pipeline_;
            }

            device_pipeline_type get_device_handle() && {
                return static_cast<DevicePipeline&&>(device_pipeline_);
            }

            constexpr int stages() const {
                return device_pipeline_.stages();
            }

            constexpr auto get_block_dim() const {
                return device_pipeline_.get_block_dim();
            }

            constexpr auto buffer_alignment() const {
                return device_pipeline_.buffer_alignment();
            }

            auto buffer_size() const {
                return device_pipeline_.buffer_size();
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_RESULT_HPP
