// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_SUGGESTED_GMMMA_HPP
#define CUBLASDX_DATABASE_SUGGESTED_GMMMA_HPP

#include "cublasdx/database/gemm/suggested/common.hpp"

namespace cublasdx {
    namespace detail {
        namespace layout_database {

            template<class ElemType, bool IsBLISKMajor, class Shape>
            CUBLASDX_HOST_DEVICE constexpr auto is_wgmma_compatible_shape(Shape) {
                auto wgmma_compatible_atoms = cute::make_tuple(
                    cute::conditional_return<IsBLISKMajor>(cute::GMMA::Layout_K_SW128_Atom<ElemType> {},
                                                           cute::GMMA::Layout_MN_SW128_Atom<ElemType> {}),
                    cute::conditional_return<IsBLISKMajor>(cute::GMMA::Layout_K_SW64_Atom<ElemType> {},
                                                           cute::GMMA::Layout_MN_SW64_Atom<ElemType> {}),
                    cute::conditional_return<IsBLISKMajor>(cute::GMMA::Layout_K_SW32_Atom<ElemType> {},
                                                           cute::GMMA::Layout_MN_SW32_Atom<ElemType> {}),
                    cute::conditional_return<IsBLISKMajor>(cute::GMMA::Layout_K_INTER_Atom<ElemType> {},
                                                           cute::GMMA::Layout_MN_INTER_Atom<ElemType> {}));

                auto is_compatible = [=](auto value, auto elem) constexpr {
                    return value || cute::C<decltype(cute::evenly_divides(Shape{}, elem.layout_b()))::value> {};
                };

                constexpr bool result = cute::fold(wgmma_compatible_atoms, cute::false_type {}, is_compatible);
                return cute::C<result> {};
            }

            template<bool IsBLISKMajor, class ElemType, class Shape>
            CUBLASDX_HOST_DEVICE constexpr auto make_wgmma_compatible_layout(Shape shape) {
                using sw128_layout = cute::conditional_t<IsBLISKMajor,
                                                         cute::GMMA::Layout_K_SW128_Atom<ElemType>,
                                                         cute::GMMA::Layout_MN_SW128_Atom<ElemType>>;
                using sw64_layout  = cute::conditional_t<IsBLISKMajor,
                                                         cute::GMMA::Layout_K_SW64_Atom<ElemType>,
                                                         cute::GMMA::Layout_MN_SW64_Atom<ElemType>>;
                using sw32_layout  = cute::conditional_t<IsBLISKMajor,
                                                         cute::GMMA::Layout_K_SW32_Atom<ElemType>,
                                                         cute::GMMA::Layout_MN_SW32_Atom<ElemType>>;
                using inter_layout = cute::conditional_t<IsBLISKMajor,
                                                         cute::GMMA::Layout_K_INTER_Atom<ElemType>,
                                                         cute::GMMA::Layout_MN_INTER_Atom<ElemType>>;

                constexpr auto tiling_order =
                    cute::conditional_return<IsBLISKMajor>(cute::GenRowMajor {}, cute::GenColMajor {});

                if constexpr (cute::evenly_divides(shape, cute::shape(sw128_layout {}))) {
                    return remove_zeros_from_layout(cute::tile_to_shape(sw128_layout {}, shape, tiling_order));
                } else if constexpr (cute::evenly_divides(shape, cute::shape(sw64_layout {}))) {
                    return remove_zeros_from_layout(cute::tile_to_shape(sw64_layout {}, shape, tiling_order));
                } else if constexpr (cute::evenly_divides(shape, cute::shape(sw32_layout {}))) {
                    return remove_zeros_from_layout(cute::tile_to_shape(sw32_layout {}, shape, tiling_order));
                } else if constexpr (cute::evenly_divides(shape, cute::shape(inter_layout {}))) {
                    return remove_zeros_from_layout(cute::tile_to_shape(inter_layout {}, shape, tiling_order));
                } else {
                    static_assert(is_wgmma_compatible_shape<ElemType, IsBLISKMajor>(shape));
                }
                CUTE_GCC_UNREACHABLE;
            }

            template<class TA, class TB, class TC, bool IsAKMajor, bool IsBKMajor>
            struct gmma_instructions {
                using type = instruction_list<>;
            };

            template<class TA, class TB, class TC, bool IsAKMajor, bool IsBKMajor>
            using gmma_instructions_t = typename gmma_instructions<TA, TB, TC, IsAKMajor, IsBKMajor>::type;

            template<bool IsAKMajor, bool IsBKMajor>
            struct gmma_instructions<half_t, half_t, half_t, IsAKMajor, IsBKMajor> {
                static constexpr auto a_major = IsAKMajor ? cute::GMMA::Major::K : cute::GMMA::Major::MN;
                static constexpr auto b_major = IsBKMajor ? cute::GMMA::Major::K : cute::GMMA::Major::MN;
                using type = instruction_list<instruction<cute::SM90_64x256x16_F16F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x192x16_F16F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x128x16_F16F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x96x16_F16F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x64x16_F16F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x32x16_F16F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x16x16_F16F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x8x16_F16F16F16_SS<a_major, b_major>, 900>>;
            };

            template<bool IsAKMajor, bool IsBKMajor>
            struct gmma_instructions<half_t, half_t, float, IsAKMajor, IsBKMajor> {
                static constexpr auto a_major = IsAKMajor ? cute::GMMA::Major::K : cute::GMMA::Major::MN;
                static constexpr auto b_major = IsBKMajor ? cute::GMMA::Major::K : cute::GMMA::Major::MN;
                using type = instruction_list<instruction<cute::SM90_64x256x16_F32F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x192x16_F32F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x128x16_F32F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x96x16_F32F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x64x16_F32F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x32x16_F32F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x16x16_F32F16F16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x8x16_F32F16F16_SS<a_major, b_major>, 900>>;
            };

            template<bool IsAKMajor, bool IsBKMajor>
            struct gmma_instructions<bfloat16_t, bfloat16_t, float, IsAKMajor, IsBKMajor> {
                static constexpr auto a_major = IsAKMajor ? cute::GMMA::Major::K : cute::GMMA::Major::MN;
                static constexpr auto b_major = IsBKMajor ? cute::GMMA::Major::K : cute::GMMA::Major::MN;
                using type = instruction_list<instruction<cute::SM90_64x256x16_F32BF16BF16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x192x16_F32BF16BF16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x128x16_F32BF16BF16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x96x16_F32BF16BF16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x64x16_F32BF16BF16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x32x16_F32BF16BF16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x16x16_F32BF16BF16_SS<a_major, b_major>, 900>,
                                              instruction<cute::SM90_64x8x16_F32BF16BF16_SS<a_major, b_major>, 900>>;
            };

            template<>
            struct gmma_instructions<tfloat32_t, tfloat32_t, float, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x8_F32TF32TF32_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x8_F32TF32TF32_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x8_F32TF32TF32_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x8_F32TF32TF32_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x8_F32TF32TF32_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x8_F32TF32TF32_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x8_F32TF32TF32_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x8_F32TF32TF32_SS_TN<>, 900>>;
            };

            template<>
            struct gmma_instructions<int8_t, int8_t, int32_t, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_S32S8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x192x32_S32S8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x128x32_S32S8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x96x32_S32S8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x64x32_S32S8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x32x32_S32S8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x16x32_S32S8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x8x32_S32S8S8_SS_TN, 900>>;
            };

            template<>
            struct gmma_instructions<uint8_t, int8_t, int32_t, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_S32U8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x192x32_S32U8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x128x32_S32U8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x96x32_S32U8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x64x32_S32U8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x32x32_S32U8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x16x32_S32U8S8_SS_TN, 900>,
                                              instruction<cute::SM90_64x8x32_S32U8S8_SS_TN, 900>>;
            };

            template<>
            struct gmma_instructions<int8_t, uint8_t, int32_t, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_S32S8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x192x32_S32S8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x128x32_S32S8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x96x32_S32S8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x64x32_S32S8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x32x32_S32S8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x16x32_S32S8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x8x32_S32S8U8_SS_TN, 900>>;
            };

            template<>
            struct gmma_instructions<uint8_t, uint8_t, int32_t, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_S32U8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x192x32_S32U8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x128x32_S32U8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x96x32_S32U8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x64x32_S32U8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x32x32_S32U8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x16x32_S32U8U8_SS_TN, 900>,
                                              instruction<cute::SM90_64x8x32_S32U8U8_SS_TN, 900>>;
            };

            template<>
            struct gmma_instructions<float_e4m3_t, float_e4m3_t, half_t, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_F16E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x32_F16E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x32_F16E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x32_F16E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x32_F16E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x32_F16E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x32_F16E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x32_F16E4M3E4M3_SS_TN<>, 900>>;
            };
            template<>
            struct gmma_instructions<float_e4m3_t, float_e4m3_t, float, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_F32E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x32_F32E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x32_F32E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x32_F32E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x32_F32E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x32_F32E4M3E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x32_F32E4M3E4M3_SS_TN<>, 900>>;
            };

            template<>
            struct gmma_instructions<float_e4m3_t, float_e5m2_t, half_t, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_F16E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x32_F16E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x32_F16E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x32_F16E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x32_F16E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x32_F16E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x32_F16E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x32_F16E4M3E5M2_SS_TN<>, 900>>;
            };
            template<>
            struct gmma_instructions<float_e4m3_t, float_e5m2_t, float, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_F32E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x32_F32E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x32_F32E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x32_F32E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x32_F32E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x32_F32E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x32_F32E4M3E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x32_F32E4M3E5M2_SS_TN<>, 900>>;
            };

            template<>
            struct gmma_instructions<float_e5m2_t, float_e4m3_t, half_t, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_F16E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x32_F16E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x32_F16E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x32_F16E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x32_F16E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x32_F16E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x32_F16E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x32_F16E5M2E4M3_SS_TN<>, 900>>;
            };
            template<>
            struct gmma_instructions<float_e5m2_t, float_e4m3_t, float, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_F32E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x32_F32E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x32_F32E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x32_F32E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x32_F32E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x32_F32E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x32_F32E5M2E4M3_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x32_F32E5M2E4M3_SS_TN<>, 900>>;
            };

            template<>
            struct gmma_instructions<float_e5m2_t, float_e5m2_t, half_t, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_F16E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x32_F16E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x32_F16E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x32_F16E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x32_F16E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x32_F16E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x32_F16E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x32_F16E5M2E5M2_SS_TN<>, 900>>;
            };
            template<>
            struct gmma_instructions<float_e5m2_t, float_e5m2_t, float, true, true> {
                using type = instruction_list<instruction<cute::SM90_64x256x32_F32E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x192x32_F32E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x128x32_F32E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x96x32_F32E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x64x32_F32E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x32x32_F32E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x16x32_F32E5M2E5M2_SS_TN<>, 900>,
                                              instruction<cute::SM90_64x8x32_F32E5M2E5M2_SS_TN<>, 900>>;
            };

            // GMMA --> Hopper only MMA instructions
            // =====================================
            template<int SM,
                     int NumWarps,
                     class GEMMShape,
                     class TA,
                     class TB,
                     class TC,
                     bool IsAKMajor,
                     bool IsBKMajor>
            struct get_best_gmma_tiled_instruction {
                using instruction_list                    = gmma_instructions_t<TA, TB, TC, IsAKMajor, IsBKMajor>;
                static constexpr bool has_full_warpgroups = NumWarps % 4 == 0;
                static constexpr auto warpgroup_tile      = get_warpgroup_tile(NumWarps);
                static constexpr int  wg_x                = cute::get<0>(warpgroup_tile);
                static constexpr int  wg_y                = cute::get<1>(warpgroup_tile);
                using WarpGroupTile = cute::Layout<cute::Shape<cute::Int<wg_x>, cute::Int<wg_y>, cute::_1>>;

                using instruction =
                    cute::conditional_t<has_full_warpgroups,
                                        typename search_mma_list<SM, WarpGroupTile, GEMMShape, instruction_list>::type,
                                        void>;
                using tile = cute::conditional_t<has_full_warpgroups, WarpGroupTile, void>;
                static constexpr instruction_type instruction_kind = instruction_type::gmma;
            };
        } // namespace layout_database
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_SUGGESTED_GMMMA_HPP
