// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_TILE_HPP
#define CUBLASDX_OPERATORS_TILE_HPP

#include <cublasdx/detail/cute/cute_tensor.hpp>

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    namespace experimental {
        template<class MMA,
                 int TileX,
                 int TileY,
                 class Permutation = cute::Tile<cute::Underscore, cute::Underscore, cute::Underscore>>
        struct Tile: public commondx::detail::operator_expression {
            using mma                   = MMA;
            using permute               = Permutation;
            static constexpr int tile_x = TileX;
            static constexpr int tile_y = TileY;

            static constexpr bool valid = not cute::is_void_v<mma>;
        };
    } // namespace experimental
} // namespace cublasdx

namespace commondx::detail {
    template<class MMA, int TileX, int TileY, class Permutation>
    struct is_operator<cublasdx::operator_type,
                       cublasdx::operator_type::experimental_tile,
                       cublasdx::experimental::Tile<MMA, TileX, TileY, Permutation>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<class MMA, int TileX, int TileY, class Permutation>
    struct get_operator_type<cublasdx::operator_type, cublasdx::experimental::Tile<MMA, TileX, TileY, Permutation>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::experimental_tile;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_TILE_HPP
