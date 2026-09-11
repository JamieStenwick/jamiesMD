// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_SUGGESTED_COMMON_HPP
#define CUBLASDX_DATABASE_SUGGESTED_COMMON_HPP

#include "cublasdx/detail/cute/cute_tensor.hpp"

namespace cublasdx {
    namespace detail {

        enum class instruction_type
        {
            fma,
            supermma,
            gmma,
            utcmma
        };

        namespace layout_database {
            CUBLASDX_HOST_DEVICE
            constexpr bool is_power_of_2(unsigned value) {
                return (value & (value - 1)) == 0;
            }

            // clz may not be accessible in compile time
            CUBLASDX_HOST_DEVICE
            constexpr unsigned log2(unsigned x) {
                return cute::bit_width(uint32_t(x)) - 1;
            }

            CUBLASDX_HOST_DEVICE
            constexpr auto get_floor_tile(unsigned value) {
                const unsigned log_2_value = log2(value);
                const unsigned value_x     = 1 << ((log_2_value + 1) / 2);
                const unsigned value_y     = value / value_x;
                return cute::make_tuple(value_x, value_y);
            }

            CUBLASDX_HOST_DEVICE
            constexpr auto get_warp_tile(unsigned num_warps) {
                return get_floor_tile(num_warps);
            }

            CUBLASDX_HOST_DEVICE
            constexpr auto get_warpgroup_tile(unsigned num_warps) {
                const unsigned warpgroups = cute::ceil_div(num_warps, 4);
                return get_floor_tile(warpgroups);
            }

            // ==========================================================
            // Get result of a metafunction or a type
            template<bool Condition, class ResultIfTrue, class MetaFunctionIfFalse>
            struct get_result;

            template<class ResultIfTrue, class MetaFunctionIfFalse>
            struct get_result<true, ResultIfTrue, MetaFunctionIfFalse> {
                using type = ResultIfTrue;
            };

            template<class ResultIfTrue, class MetaFunctionIfFalse>
            struct get_result<false, ResultIfTrue, MetaFunctionIfFalse> {
                using type = typename MetaFunctionIfFalse::type;
            };

            template<bool Condition, class ResultIfTrue, class MetaFunctionIfFalse>
            using get_result_t = typename get_result<Condition, ResultIfTrue, MetaFunctionIfFalse>::type;

            using warp_thread_layout_supermma = cute::Shape<cute::_8, cute::_4>;

            template<class Instruction, unsigned SMMin, unsigned SMMax = 0>
            struct instruction {
                using type                       = Instruction;
                static constexpr unsigned sm_min = SMMin;
                static constexpr unsigned sm_max = SMMax;
            };

            template<class... Instructions>
            struct instruction_list {
            };

            // ==========================================================
            // Return first (best) instruction fulfilling the SM criteria
            template<int SM, class ExecTile, class GEMMShape, class List>
            struct search_mma_list {
                using type = void;
            };

            template<int SM, class ExecTile, class GEMMShape, class Instruction, unsigned SMMin, unsigned SMMax, class... Elems>
            struct search_mma_list<SM,
                                   ExecTile,
                                   GEMMShape,
                                   instruction_list<instruction<Instruction, SMMin, SMMax>, Elems...>> {
                using instruction_shape = typename cute::MMA_Traits<Instruction>::Shape_MNK;
                static constexpr int warp_tiled_m =
                    cute::get<0>(cute::shape(ExecTile {})) * cute::get<0>(instruction_shape {});
                static constexpr int warp_tiled_n =
                    cute::get<1>(cute::shape(ExecTile {})) * cute::get<1>(instruction_shape {});
                static constexpr int warp_tiled_k =
                    cute::get<2>(cute::shape(ExecTile {})) * cute::get<2>(instruction_shape {});
                using warp_tiled_instruction_shape =
                    cute::Shape<cute::Int<warp_tiled_m>, cute::Int<warp_tiled_n>, cute::Int<warp_tiled_k>>;
                static constexpr bool condition =
                    (SM >= SMMin) and (SMMax == 0 or SM <= SMMax) and 
                    cute::evenly_divides(GEMMShape {}, warp_tiled_instruction_shape {});

                using type = get_result_t<condition,
                                          Instruction,
                                          search_mma_list<SM, ExecTile, GEMMShape, instruction_list<Elems...>>>;
            };

            // ==========================================================

            struct null_instruction {
            };

        } // namespace layout_database
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_SUGGESTED_COMMON_HPP
