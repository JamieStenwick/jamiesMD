// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_OPERATOR_TYPE_HPP
#define CUBLASDX_OPERATORS_OPERATOR_TYPE_HPP

#include "commondx/detail/expressions.hpp"

namespace cublasdx {
    enum class operator_type
    {
        size,
        precision,
        type,
        function,
        transpose_mode,
        sm,
        alignment,
        arrangement,
        ld,
        streaming,
        with_pipeline,
        required_mantissa_bits,
        // execution
        block,
        thread,
        // block only
        block_dim,
        static_block_dim,
        // TRSM-specific
        batches_per_block,
        fill_mode,
        diag,
        side,
        // experimental
        experimental_tile
    };
} // namespace cublasdx

#endif // CUBLASDX_OPERATORS_OPERATOR_TYPE_HPP
