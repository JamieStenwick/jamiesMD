// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_OPERATOR_TYPE_HPP
#define CUSOLVERDX_OPERATORS_OPERATOR_TYPE_HPP

namespace cusolverdx {
    enum class operator_type
    {
        size,
        function,
        precision,
        type,
        fill_mode, // only used by trsm, potrf, potrs, posv, heev, hegv
        leading_dimension,
        arrangement,
        transpose,  // only used by trsm, getrs, gesv, unmqr, unmlq, gels
        diag,       // only used by trsm
        side,       // only used by trsm, unmqr, unmlq
        job,        // only used by htev, heev, hegv, bdsvd, gesvd
        eig_type, // only used by hegst, hegv
        sm,
        // execution
        thread,
        block,
        cluster,
        // block and cluster only
        block_dim,
        // block only
        batches_per_block,
        // cluster only
        blocks_per_cluster,
        tile_size
    };
} // namespace cusolverdx

#endif // CUSOLVERDX_OPERATORS_OPERATOR_TYPE_HPP
