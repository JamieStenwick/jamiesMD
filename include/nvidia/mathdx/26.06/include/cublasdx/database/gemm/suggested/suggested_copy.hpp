// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_SUGGESTED_COPY_HPP
#define CUBLASDX_DATABASE_SUGGESTED_COPY_HPP

#include "cublasdx/database/gemm/suggested/common.hpp"

namespace cublasdx {
    namespace detail {
        namespace layout_database {
            // List of necessary copy operations
            // =================================
            using auto_vectorizing_copy = cute::AutoVectorizingCopyWithAssumedAlignment<128>;

            template<bool IsKMajor, class ElemType, class = void>
            struct ldsm_copy_instructions {
                using type = instruction_list<>;
            };

            template<bool IsKMajor, class ElemType, class = void>
            struct stsm_copy_instructions {
                using type = instruction_list<>;
            };

            template<class ElemType>
            struct is_ldsm_n_compatible_type {
                static constexpr bool value =
                    cute::is_same_v<ElemType, int8_t> or cute::is_same_v<ElemType, uint8_t> or
                    cute::is_same_v<ElemType, float_e4m3_t> or cute::is_same_v<ElemType, float_e5m2_t> or
                    cute::is_same_v<ElemType, half_t> or cute::is_same_v<ElemType, bfloat16_t> or
                    cute::is_same_v<ElemType, tfloat32_t>;
            };

            template<class ElemType>
            struct is_ldsm_t_compatible_type {
                static constexpr bool value =
                    cute::is_same_v<ElemType, half_t> or cute::is_same_v<ElemType, bfloat16_t>;
            };

            template<class ElemType>
            struct ldsm_copy_instructions<true,
                                          ElemType,
                                          cute::enable_if_t<is_ldsm_n_compatible_type<ElemType>::value>> {
                using type = instruction_list<instruction<cute::SM75_U32x4_LDSM_N, 750>,
                                              instruction<cute::SM75_U32x2_LDSM_N, 750>,
                                              instruction<cute::SM75_U32x1_LDSM_N, 750>>;
            };

            template<class ElemType>
            struct ldsm_copy_instructions<false,
                                          ElemType,
                                          cute::enable_if_t<is_ldsm_t_compatible_type<ElemType>::value>> {
                using type = instruction_list<instruction<cute::SM75_U16x8_LDSM_T, 750>,
                                              instruction<cute::SM75_U16x4_LDSM_T, 750>,
                                              instruction<cute::SM75_U16x2_LDSM_T, 750>>;
            };

            // Warning: STSM cannot be always used, only when output precision matches input precision
            // in size and alignment
            template<class ElemType>
            struct stsm_copy_instructions<true,
                                          ElemType,
                                          cute::enable_if_t<is_ldsm_t_compatible_type<ElemType>::value>> {
                using type = instruction_list<instruction<cute::SM90_U32x4_STSM_N, 900>,
                                              instruction<cute::SM90_U32x2_STSM_N, 900>,
                                              instruction<cute::SM90_U32x1_STSM_N, 900>>;
            };

            template<class ElemType>
            struct stsm_copy_instructions<false,
                                          ElemType,
                                          cute::enable_if_t<is_ldsm_t_compatible_type<ElemType>::value>> {
                using type = instruction_list<instruction<cute::SM90_U16x8_STSM_T, 900>,
                                              instruction<cute::SM90_U16x4_STSM_T, 900>,
                                              instruction<cute::SM90_U16x2_STSM_T, 900>>;
            };

            template<bool IsKMajor, class ElemType>
            using ldsm_copy_instructions_t = typename ldsm_copy_instructions<IsKMajor, ElemType>::type;

            template<bool IsKMajor, class ElemType>
            using stsm_copy_instructions_t = typename stsm_copy_instructions<IsKMajor, ElemType>::type;


            template<int SM, int AtomBytes, int MaxMult, class List>
            struct search_ldst_list {
                using type = void;
            };

            template<int SM, int AtomBytes, int MaxMult, class Instruction, unsigned SMMin, unsigned SMMax, class... Elems>
            struct search_ldst_list<SM,
                                    AtomBytes,
                                    MaxMult,
                                    instruction_list<instruction<Instruction, SMMin, SMMax>, Elems...>> {
                static constexpr int copy_size_bytes =
                    cute::size(typename cute::Copy_Traits<Instruction>::RefLayout {}) / cute::sizeof_bits_v<char>;

                // Use cute::ceil_div to avoid division by zero, but verify divisibility manually
                static constexpr bool tiling_factor_divisibility = copy_size_bytes % AtomBytes == 0;
                static constexpr int  required_tiling_factor     = cute::ceil_div(copy_size_bytes, AtomBytes);

                static constexpr bool condition =
                    (SM >= SMMin) and (SMMax == 0 or SM <= SMMax) and tiling_factor_divisibility and ((MaxMult % required_tiling_factor) == 0);
                using type = get_result_t<condition,
                                          Instruction,
                                          search_ldst_list<SM, AtomBytes, MaxMult, instruction_list<Elems...>>>;
            };

            template<int SM, class TileShape, class AtomShape, class ElemType, bool IsKMajor>
            struct get_best_ldst_instruction {
                static constexpr int atom_bytes =
                    cute::get<1>(AtomShape {}) * cute::get<2>(AtomShape {}) * sizeof(ElemType);
                static constexpr int tile_m = cute::get<0>(TileShape {});
                // AtomShape needs to be <WarpM, TileM, TileN>
                static constexpr int tiled_mma_m = cute::get<0>(AtomShape {}) * cute::get<1>(AtomShape {});

                static constexpr bool is_predicated = tile_m % tiled_mma_m != 0;
                static constexpr bool is_complex    = has_complex_interface<ElemType>();
                static constexpr bool is_divisible  = tile_m % tiled_mma_m == 0;
                static constexpr bool unsupported   = is_predicated or is_complex or not is_divisible;
                static constexpr int  max_mult      = tile_m / tiled_mma_m;
                using load_type                     = get_result_t<
                                        unsupported,
                                        void,
                                        search_ldst_list<SM, atom_bytes, max_mult, ldsm_copy_instructions_t<IsKMajor, ElemType>>>;
                using store_type = get_result_t<
                    unsupported,
                    void,
                    search_ldst_list<SM, atom_bytes, max_mult, stsm_copy_instructions_t<IsKMajor, ElemType>>>;
            };

            template<int SM, class TileShape, class AtomShape, class ElemType, bool IsKMajor>
            using get_best_ldsm_instruction_t =
                typename get_best_ldst_instruction<SM, TileShape, AtomShape, ElemType, IsKMajor>::load_type;

            template<int SM, class TileShape, class AtomShape, class ElemType, bool IsKMajor>
            using get_best_stsm_instruction_t =
                typename get_best_ldst_instruction<SM, TileShape, AtomShape, ElemType, IsKMajor>::store_type;
        } // namespace layout_database
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_SUGGESTED_COPY_HPP
