// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_SUGGESTED_PORTABLE_MMA_HPP
#define CUBLASDX_DATABASE_SUGGESTED_PORTABLE_MMA_HPP

#include "cublasdx/database/gemm/suggested/common.hpp"

namespace cublasdx {
    namespace detail {
        namespace layout_database {

            // ==========================================================

            template<class TA, class TB, class TC>
            struct supermma_instructions {
                using type = instruction_list<>;
            };

            // ==========================================================
            // Lists of superMMA instructions sorted by size (preference)

            template<>
            struct supermma_instructions<int8_t, int8_t, int32_t> {
                using type = instruction_list<instruction<cute::SM80_16x8x32_S32S8S8S32_TN, 800>,
                                              instruction<cute::SM80_16x8x16_S32S8S8S32_TN, 800>,
                                              instruction<cute::SM80_8x8x16_S32S8S8S32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<int8_t, uint8_t, int32_t> {
                using type = instruction_list<instruction<cute::SM80_16x8x32_S32S8U8S32_TN, 800>,
                                              instruction<cute::SM80_16x8x16_S32S8U8S32_TN, 800>,
                                              instruction<cute::SM80_8x8x16_S32S8U8S32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<uint8_t, int8_t, int32_t> {
                using type = instruction_list<instruction<cute::SM80_16x8x32_S32U8S8S32_TN, 800>,
                                              instruction<cute::SM80_16x8x16_S32U8S8S32_TN, 800>,
                                              instruction<cute::SM80_8x8x16_S32U8S8S32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<uint8_t, uint8_t, int32_t> {
                using type = instruction_list<instruction<cute::SM80_16x8x32_S32U8U8S32_TN, 800>,
                                              instruction<cute::SM80_16x8x16_S32U8U8S32_TN, 800>,
                                              instruction<cute::SM80_8x8x16_S32U8U8S32_TN, 800>>;
            };


            template<>
            struct supermma_instructions<float_e4m3_t, float_e4m3_t, float> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_F32E4M3E4M3F32_TN, 890>
                                              ,
                                              instruction<cublasdx::detail::SM89_16x8x16_F32E4M3E4M3F32_TN, 890>
                                              >;
            };

            template<>
            struct supermma_instructions<float_e4m3_t, float_e5m2_t, float> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_F32E4M3E5M2F32_TN, 890>
                                              ,
                                              instruction<cublasdx::detail::SM89_16x8x16_F32E4M3E5M2F32_TN, 890>
                                              >;
            };

            template<>
            struct supermma_instructions<float_e5m2_t, float_e4m3_t, float> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_F32E5M2E4M3F32_TN, 890>
                                              ,
                                              instruction<cublasdx::detail::SM89_16x8x16_F32E5M2E4M3F32_TN, 890>
                                              >;
            };

            template<>
            struct supermma_instructions<float_e5m2_t, float_e5m2_t, float> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_F32E5M2E5M2F32_TN, 890>
                                              ,
                                              instruction<cublasdx::detail::SM89_16x8x16_F32E5M2E5M2F32_TN, 890>
                                              >;
            };



            template<>
            struct supermma_instructions<float_e4m3_t, float_e4m3_t, cute::half_t> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_F16E4M3E4M3F16_TN, 890>,
                                              instruction<cublasdx::detail::SM89_16x8x16_F16E4M3E4M3F16_TN, 890>>;
            };

            template<>
            struct supermma_instructions<float_e4m3_t, float_e5m2_t, cute::half_t> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_F16E4M3E5M2F16_TN, 890>,
                                              instruction<cublasdx::detail::SM89_16x8x16_F16E4M3E5M2F16_TN, 890>>;
            };

            template<>
            struct supermma_instructions<float_e5m2_t, float_e4m3_t, cute::half_t> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_F16E5M2E4M3F16_TN, 890>,
                                              instruction<cublasdx::detail::SM89_16x8x16_F16E5M2E4M3F16_TN, 890>>;
            };

            template<>
            struct supermma_instructions<float_e5m2_t, float_e5m2_t, cute::half_t> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_F16E5M2E5M2F16_TN, 890>,
                                              instruction<cublasdx::detail::SM89_16x8x16_F16E5M2E5M2F16_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cute::half_t, cute::half_t, cute::half_t> {
                using type =
                    instruction_list<instruction<cute::SM80_16x8x16_F16F16F16F16_TN, 800>, instruction<cublasdx::detail::SM75_16x8x8_F16F16F16F16_TN, 750>, >;
            };

            template<>
            struct supermma_instructions<cute::half_t, cute::half_t, float> {
                using type = instruction_list<instruction<cute::SM80_16x8x16_F32F16F16F32_TN, 800>,
                                              instruction<cublasdx::detail::SM75_16x8x8_F32F16F16F32_TN, 750>>;
            };

            template<>
            struct supermma_instructions<bfloat16_t, bfloat16_t, float> {
                using type = instruction_list<instruction<cute::SM80_16x8x16_F32BF16BF16F32_TN, 800>,
                                              instruction<cute::SM80_16x8x8_F32BF16BF16F32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<tfloat32_t, tfloat32_t, float> {
                using type = instruction_list<instruction<cute::SM80_16x8x8_F32TF32TF32F32_TN, 800>,
                                              instruction<cute::SM80_16x8x4_F32TF32TF32F32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<double, double, double> {
                using type = instruction_list<instruction<cute::SM90_16x8x4_F64F64F64F64_TN, 900, 900>, // limited to Hopper
                                              instruction<cute::SM90_16x8x8_F64F64F64F64_TN, 900, 900>, // limited to Hopper
                                              instruction<cute::SM90_16x8x16_F64F64F64F64_TN, 900, 900>, // limited to Hopper
                                              instruction<cute::SM80_8x8x4_F64F64F64F64_TN, 800>>;
            };

            // Complex types
            template<>
            struct supermma_instructions<cutlass::complex<int8_t>,
                                         cutlass::complex<int8_t>,
                                         cutlass::complex<int32_t>> {
                using type = instruction_list<instruction<cublasdx::detail::SM80_16x8x32_CS32CS8CS8CS32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_16x8x16_CS32CS8CS8CS32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_8x8x16_CS32CS8CS8CS32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<int8_t>,
                                         cutlass::complex<uint8_t>,
                                         cutlass::complex<int32_t>> {
                using type = instruction_list<instruction<cublasdx::detail::SM80_16x8x32_CS32CS8CU8CS32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_16x8x16_CS32CS8CU8CS32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_8x8x16_CS32CS8CU8CS32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<uint8_t>,
                                         cutlass::complex<int8_t>,
                                         cutlass::complex<int32_t>> {
                using type = instruction_list<instruction<cublasdx::detail::SM80_16x8x32_CS32CU8CS8CS32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_16x8x16_CS32CU8CS8CS32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_8x8x16_CS32CU8CS8CS32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<uint8_t>,
                                         cutlass::complex<uint8_t>,
                                         cutlass::complex<int32_t>> {
                using type = instruction_list<instruction<cublasdx::detail::SM80_16x8x32_CS32CU8CU8CS32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_16x8x16_CS32CU8CU8CS32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_8x8x16_CS32CU8CU8CS32_TN, 800>>;
            };


            template<>
            struct supermma_instructions<cutlass::complex<float_e4m3_t>,
                                         cutlass::complex<float_e4m3_t>,
                                         cutlass::complex<float>> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_C32CE4M3CE4M3C32_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<float_e4m3_t>,
                                         cutlass::complex<float_e5m2_t>,
                                         cutlass::complex<float>> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_C32CE4M3CE5M2C32_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<float_e5m2_t>,
                                         cutlass::complex<float_e4m3_t>,
                                         cutlass::complex<float>> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_C32CE5M2CE4M3C32_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<float_e5m2_t>,
                                         cutlass::complex<float_e5m2_t>,
                                         cutlass::complex<float>> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_C32CE5M2CE5M2C32_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<float_e4m3_t>,
                                         cutlass::complex<float_e4m3_t>,
                                         cutlass::complex<cute::half_t>> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_C16CE4M3CE4M3C16_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<float_e4m3_t>,
                                         cutlass::complex<float_e5m2_t>,
                                         cutlass::complex<cute::half_t>> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_C16CE4M3CE5M2C16_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<float_e5m2_t>,
                                         cutlass::complex<float_e4m3_t>,
                                         cutlass::complex<cute::half_t>> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_C16CE5M2CE4M3C16_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<float_e5m2_t>,
                                         cutlass::complex<float_e5m2_t>,
                                         cutlass::complex<cute::half_t>> {
                using type = instruction_list<instruction<cublasdx::detail::SM89_16x8x32_C16CE5M2CE5M2C16_TN, 890>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<half_t>> {
                using type =
                    instruction_list<instruction<cublasdx::detail::SM80_16x8x16_C16C16C16C16_TN, 800>, instruction<cublasdx::detail::SM75_16x8x8_C16C16C16C16_TN, 750>, >;
            };

            template<>
            struct supermma_instructions<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<float>> {
                using type = instruction_list<instruction<cublasdx::detail::SM80_16x8x16_C32C16C16C32_TN, 800>,
                                              instruction<cublasdx::detail::SM75_16x8x8_C32C16C16C32_TN, 750>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<bfloat16_t>,
                                         cutlass::complex<bfloat16_t>,
                                         cutlass::complex<float>> {
                using type = instruction_list<instruction<cublasdx::detail::SM80_16x8x16_C32BC16BC16C32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_16x8x8_C32BC16BC16C32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<tfloat32_t>,
                                         cutlass::complex<tfloat32_t>,
                                         cutlass::complex<float>> {
                using type = instruction_list<instruction<cublasdx::detail::SM80_16x8x8_C32TC32TC32C32_TN, 800>,
                                              instruction<cublasdx::detail::SM80_16x8x4_C32TC32TC32C32_TN, 800>>;
            };

            template<>
            struct supermma_instructions<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>> {
                using type = instruction_list<instruction<cute::SM90_16x8x4_C64C64C64C64_TN, 900, 900>, // limited to Hopper
                                              instruction<cute::SM90_16x8x8_C64C64C64C64_TN, 900, 900>, // limited to Hopper
                                              instruction<cute::SM90_16x8x16_C64C64C64C64_TN, 900, 900>, // limited to Hopper
                                              instruction<cute::SM80_8x8x4_C64C64C64C64_TN, 800>>;
            };

            template<class TA, class TB, class TC>
            using supermma_instructions_t = typename supermma_instructions<TA, TB, TC>::type;

            template<int SM, int NumWarps, class GEMMShape, class TA, class TB, class TC>
            struct get_best_supermma_tiled_instruction {
                using instruction_list          = supermma_instructions_t<TA, TB, TC>;
                static constexpr auto warp_tile = get_warp_tile(NumWarps);
                static constexpr int  warp_x    = cute::get<0>(warp_tile);
                static constexpr int  warp_y    = cute::get<1>(warp_tile);
                using col_major_tile            = cute::Shape<cute::Int<warp_x>, cute::Int<warp_y>, cute::_1>;

                static constexpr bool make_warp_tile_row_major = cute::size<1>(GEMMShape {}) >
                                                                 cute::size<0>(GEMMShape {});
                static constexpr int warp_tile_row = cute::get < make_warp_tile_row_major ? 1 : 0 > (col_major_tile {});
                static constexpr int warp_tile_col = cute::get < make_warp_tile_row_major ? 0 : 1 > (col_major_tile {});

                using search_tile =
                    cute::Layout<cute::Shape<cute::Int<warp_tile_row>, cute::Int<warp_tile_col>, cute::_1>>;

                using instruction = typename search_mma_list<SM, search_tile, GEMMShape, instruction_list>::type;
                using tile        = cute::conditional_t<cute::is_void_v<instruction>, void, search_tile>;
                static constexpr instruction_type instruction_kind = instruction_type::supermma;
            };

        } // namespace layout_database
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_SUGGESTED_PORTABLE_MMA_HPP