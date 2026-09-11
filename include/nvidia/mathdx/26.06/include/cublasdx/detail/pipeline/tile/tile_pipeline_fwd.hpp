// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_FWD_HPP
#define CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_FWD_HPP

#include "cublasdx/detail/copy.hpp"
#include "cublasdx/detail/pipeline/accumulator_mode.hpp"

namespace cublasdx::detail {

    template<unsigned, result_storage, copy_kind, class, bool, int, int, class, class, class, class, class, typename, class, typename>
    struct rmem_unified_pipeline;

    template<unsigned, result_storage, copy_kind, class, bool, int, int, class, class, class, class, class, typename, class, typename>
    struct rmem_specialized_pipeline;

    template<unsigned, result_storage, copy_kind, class, bool, int, class, class, class, class, class, typename, class, typename>
    struct tmem_unified_pipeline;

    template<unsigned, result_storage, copy_kind, class, bool, int, class, class, class, class, class, typename, class, typename>
    struct tmem_specialized_pipeline;

    template<class BLAS>
    struct tmem_specialized_execution_model;

} // namespace cublasdx::detail

#endif // CUBLASDX_DETAIL_PIPELINE_TILE_PIPELINE_FWD_HPP
