// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_ACCUMULATOR_MODE_HPP
#define CUBLASDX_DETAIL_PIPELINE_ACCUMULATOR_MODE_HPP

#include <cute/container/tuple.hpp>

#include "cublasdx/detail/pipeline/barriers/barrier_view.hpp"

// Allow for libmathdx overloading of function names
#ifndef CUBLASDX_OVERLOAD_DEVICE_PIPELINE_CREATION
#define CUBLASDX_MAKE_TMA_ATOM cute::make_tma_atom
#define CUBLASDX_PIPELINE_EXECUTION __host__
#define CUBLASDX_DEVICE_PIPELINE_EXECUTION __host__ __device__ __forceinline__
#else // CUBLASDX_OVERLOAD_DEVICE_PIPELINE_CREATION
#ifndef CUBLASDX_MAKE_TMA_ATOM
#error "CUBLASDX_MAKE_TMA_ATOM undefined"
#endif // CUBLASDX_OVERLOAD_DEVICE_PIPELINE_CREATION

#ifndef CUBLASDX_PIPELINE_EXECUTION
#error "CUBLASDX_PIPELINE_EXECUTION undefined"
#endif // CUBLASDX_PIPELINE_EXECUTION

#ifndef CUBLASDX_DEVICE_PIPELINE_EXECUTION
#error "CUBLASDX_DEVICE_PIPELINE_EXECUTION undefined"
#endif
#endif // CUBLASDX_DEVICE_PIPELINE_EXECUTION

namespace cublasdx {

    enum class accumulator_mode
    {
        internal_accumulator,
        reusable_accumulator,
        external_accumulator
    };

    using result_storage = accumulator_mode;

    using pipeline_stage_scratch_t = cute::tuple<detail::barrier_storage_t, detail::barrier_storage_t>;

    inline constexpr accumulator_mode internal_accumulator   = accumulator_mode::internal_accumulator;
    inline constexpr accumulator_mode reusable_accumulator   = accumulator_mode::reusable_accumulator;
    inline constexpr accumulator_mode external_accumulator   = accumulator_mode::external_accumulator;
    inline constexpr accumulator_mode internal_accumulation  = internal_accumulator;
    inline constexpr accumulator_mode external_accumulation  = external_accumulator;

    namespace detail {
        template<auto...>
        inline constexpr bool always_false_v = false;

        // Ensure users are informed about differences between external/reusable - extra 
        // synchronization point is required in non-epilogue path. 
        template<accumulator_mode Mode>
        struct validate_accumulator_mode {
            static_assert(
                Mode != external_accumulator,
                "cublasdx::external_accumulator/external_accumulation is now cublasdx::reusable_accumulator. "
                "Use cublasdx::reusable_accumulator when accumulating multiple times into the same accumulator, "
                "then call tile_pipeline.finish_accumulation() after the final tile_pipeline.execute(accumulator) before reading the "
                "accumulator. Call finish_accumulation() with the same thread scope as execute(accumulator); do not "
                "guard it with accumulator.is_thread_active(), because internal warp-specialization may require broader "
                "participation. tile_pipeline.epilogue(accumulator, functor) remains compatible and calls "
                "finish_accumulation() internally.");
            static constexpr accumulator_mode value = Mode;
        };
    } // namespace detail

} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_ACCUMULATOR_MODE_HPP
