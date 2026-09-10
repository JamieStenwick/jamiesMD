// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_SUGGESTED_INSTRUCTIONS_HPP
#define CUBLASDX_SUGGESTED_INSTRUCTIONS_HPP

#include "cublasdx/detail/cute/cute_tensor.hpp"
#include "cublasdx/database/gemm/mma_atoms.hpp"

#include "cublasdx/database/gemm/suggested/suggested_supermma.hpp"
#include "cublasdx/database/gemm/suggested/suggested_copy.hpp"
#include "cublasdx/database/gemm/suggested/suggested_gmma.hpp"
#include "cublasdx/database/gemm/suggested/suggested_utcmma.hpp"

namespace cublasdx {
    namespace detail {
        namespace layout_database {

            // =================================
            // Public interface
            // =================================

            // Turing+ superMMA
            template<int SM, int NumWarps, class GEMMShape, class TA, class TB, class TC>
            using get_best_supermma_instruction_t =
                typename get_best_supermma_tiled_instruction<SM, NumWarps, GEMMShape, TA, TB, TC>::instruction;

            template<int SM, int NumWarps, class GEMMShape, class TA, class TB, class TC>
            using get_best_supermma_tile_t =
                typename get_best_supermma_tiled_instruction<SM, NumWarps, GEMMShape, TA, TB, TC>::tile;

            // Hopper GMMA
            template<int SM,
                     int NumWarps,
                     class GEMMShape,
                     class TA,
                     class TB,
                     class TC,
                     bool IsAKMajor,
                     bool IsBKMajor>
            using get_best_gmma_instruction_t =
                typename get_best_gmma_tiled_instruction<SM, NumWarps, GEMMShape, TA, TB, TC, IsAKMajor, IsBKMajor>::
                    instruction;

            template<int SM,
                     int NumWarps,
                     class GEMMShape,
                     class TA,
                     class TB,
                     class TC,
                     bool IsAKMajor,
                     bool IsBKMajor>
            using get_best_gmma_tile_t =
                typename get_best_gmma_tiled_instruction<SM, NumWarps, GEMMShape, TA, TB, TC, IsAKMajor, IsBKMajor>::
                    tile;

            // Blackwell UTCMMA
            template<int SM,
                     int NumWarps,
                     class GEMMShape,
                     class TA,
                     class TB,
                     class TC,
                     bool IsAKMajor,
                     bool IsBKMajor>
            using get_best_utcmma_instruction_t =
                typename get_best_utcmma_tiled_instruction<SM, NumWarps, GEMMShape, TA, TB, TC, IsAKMajor, IsBKMajor>::
                    instruction;

            template<int SM,
                     int NumWarps,
                     class GEMMShape,
                     class TA,
                     class TB,
                     class TC,
                     bool IsAKMajor,
                     bool IsBKMajor>
            using get_best_utcmma_tile_t =
                typename get_best_utcmma_tiled_instruction<SM, NumWarps, GEMMShape, TA, TB, TC, IsAKMajor, IsBKMajor>::
                    tile;

        } // namespace layout_database
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_SUGGESTED_INSTRUCTIONS_HPP
