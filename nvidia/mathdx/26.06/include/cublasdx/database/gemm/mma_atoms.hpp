// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP
#define CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP

#include "cublasdx/detail/cute/cute_tensor.hpp"

namespace cublasdx {
    namespace detail {

        namespace cute_backend {
            template<class T>
            struct map_precision_type {
                using precision_type = T;
            };
            template<typename T>
            struct map_precision_type<cutlass::complex<T>> {
                using precision_type = T;
            };
            template<typename T>
            using map_precision_type_t = typename map_precision_type<T>::precision_type;

            // This list is currently incomplete,for extensive testing add all cute::MMA types here
            enum class mma_atom
            {
                // Universal FMA
                universal_fma,

                // FP32
                SM100_2x1x1_F32F32F32F32,
                SM100_1x2x1_F32F32F32F32,

                // FP16
                SM75_16x8x8_F16F16F16F16_TN_CUBLASDX,
                SM75_16x8x8_C16C16C16C16_TN_CUBLASDX,
                SM80_16x8x16_F16F16F16F16_TN,
                SM80_16x8x16_C16C16C16C16_TN_CUBLASDX,

                // FP16 x FP32
                SM75_16x8x8_F32F16F16F32_TN_CUBLASDX,
                SM75_16x8x8_C32C16C16C32_TN_CUBLASDX,
                SM80_16x8x16_F32F16F16F32_TN,
                SM80_16x8x16_C32C16C16C32_TN_CUBLASDX,

                // BF16
                SM80_16x8x8_F32BF16BF16F32_TN,
                SM80_16x8x8_C32BC16BC16C32_TN_CUBLASDX,
                SM80_16x8x16_F32BF16BF16F32_TN,
                SM80_16x8x16_C32BC16BC16C32_TN_CUBLASDX,

                // TF32
                SM80_16x8x4_F32TF32TF32F32_TN,
                SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX,
                SM80_16x8x8_F32TF32TF32F32_TN,
                SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX,

                // INT8
                SM80_8x8x16_S32S8S8S32_TN,
                SM80_8x8x16_S32S8U8S32_TN,
                SM80_8x8x16_S32U8S8S32_TN,
                SM80_8x8x16_S32U8U8S32_TN,
                SM80_16x8x16_S32S8S8S32_TN,
                SM80_16x8x16_S32S8U8S32_TN,
                SM80_16x8x16_S32U8S8S32_TN,
                SM80_16x8x16_S32U8U8S32_TN,
                SM80_16x8x32_S32S8S8S32_TN,
                SM80_16x8x32_S32S8U8S32_TN,
                SM80_16x8x32_S32U8S8S32_TN,
                SM80_16x8x32_S32U8U8S32_TN,
                // Complex INT8
                SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX,
                SM80_8x8x16_CS32CS8CU8CS32_TN_CUBLASDX,
                SM80_8x8x16_CS32CU8CS8CS32_TN_CUBLASDX,
                SM80_8x8x16_CS32CU8CU8CS32_TN_CUBLASDX,
                SM80_16x8x16_CS32CS8CS8CS32_TN_CUBLASDX,
                SM80_16x8x16_CS32CS8CU8CS32_TN_CUBLASDX,
                SM80_16x8x16_CS32CU8CS8CS32_TN_CUBLASDX,
                SM80_16x8x16_CS32CU8CU8CS32_TN_CUBLASDX,
                SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX,
                SM80_16x8x32_CS32CS8CU8CS32_TN_CUBLASDX,
                SM80_16x8x32_CS32CU8CS8CS32_TN_CUBLASDX,
                SM80_16x8x32_CS32CU8CU8CS32_TN_CUBLASDX,
                // FP8
                SM89_16x8x32_F32E4M3E4M3F32_TN_CUBLASDX,
                SM89_16x8x32_F32E4M3E5M2F32_TN_CUBLASDX,
                SM89_16x8x32_F32E5M2E4M3F32_TN_CUBLASDX,
                SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX,
                SM89_16x8x32_F16E4M3E4M3F16_TN_CUBLASDX,
                SM89_16x8x32_F16E4M3E5M2F16_TN_CUBLASDX,
                SM89_16x8x32_F16E5M2E4M3F16_TN_CUBLASDX,
                SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX,

                SM89_16x8x16_F32E4M3E4M3F32_TN_CUBLASDX,
                SM89_16x8x16_F32E4M3E5M2F32_TN_CUBLASDX,
                SM89_16x8x16_F32E5M2E4M3F32_TN_CUBLASDX,
                SM89_16x8x16_F32E5M2E5M2F32_TN_CUBLASDX,
                SM89_16x8x16_F16E4M3E4M3F16_TN_CUBLASDX,
                SM89_16x8x16_F16E4M3E5M2F16_TN_CUBLASDX,
                SM89_16x8x16_F16E5M2E4M3F16_TN_CUBLASDX,
                SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX,

                // Complex FP8
                SM89_16x8x32_C32CE4M3CE4M3C32_TN_CUBLASDX,
                SM89_16x8x32_C32CE4M3CE5M2C32_TN_CUBLASDX,
                SM89_16x8x32_C32CE5M2CE4M3C32_TN_CUBLASDX,
                SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX,
                SM89_16x8x32_C16CE4M3CE4M3C16_TN_CUBLASDX,
                SM89_16x8x32_C16CE4M3CE5M2C16_TN_CUBLASDX,
                SM89_16x8x32_C16CE5M2CE4M3C16_TN_CUBLASDX,
                SM89_16x8x32_C16CE5M2CE5M2C16_TN_CUBLASDX,

                SM89_16x8x16_C32CE4M3CE4M3C32_TN_CUBLASDX,
                SM89_16x8x16_C32CE4M3CE5M2C32_TN_CUBLASDX,
                SM89_16x8x16_C32CE5M2CE4M3C32_TN_CUBLASDX,
                SM89_16x8x16_C32CE5M2CE5M2C32_TN_CUBLASDX,
                SM89_16x8x16_C16CE4M3CE4M3C16_TN_CUBLASDX,
                SM89_16x8x16_C16CE4M3CE5M2C16_TN_CUBLASDX,
                SM89_16x8x16_C16CE5M2CE4M3C16_TN_CUBLASDX,
                SM89_16x8x16_C16CE5M2CE5M2C16_TN_CUBLASDX,

                // FP64
                SM80_8x8x4_F64F64F64F64_TN,
                SM80_8x8x4_C64C64C64C64_TN,
                SM90_16x8x4_F64F64F64F64_TN,
                SM90_16x8x8_F64F64F64F64_TN,
                SM90_16x8x16_F64F64F64F64_TN,
                SM90_16x8x4_C64C64C64C64_TN,
                SM90_16x8x8_C64C64C64C64_TN,
                SM90_16x8x16_C64C64C64C64_TN
            };

            template<typename AType, typename BType, typename CType, unsigned int SM>
            constexpr mma_atom get_default_mma() {
                [[maybe_unused]] constexpr bool is_complex_type = cutlass::is_complex<AType>::value;
                using A_prec_t                                  = map_precision_type_t<AType>;
                using B_prec_t                                  = map_precision_type_t<BType>;
                using C_prec_t                                  = map_precision_type_t<CType>;

                mma_atom ret = mma_atom::universal_fma;
                if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                              cute::tuple<float, float, float>>) {
                    if constexpr (SM == 1000 or SM == 1030) {
                        ret = is_complex_type ? mma_atom::universal_fma : mma_atom::SM100_2x1x1_F32F32F32F32;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<half_t, half_t, half_t>>) {
                    // warning: checks on SM should be in the decreasing order
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x16_C16C16C16C16_TN_CUBLASDX
                                              : mma_atom::SM80_16x8x16_F16F16F16F16_TN;
                    } else if constexpr (SM >= 750) {
                        ret = is_complex_type ? mma_atom::SM75_16x8x8_C16C16C16C16_TN_CUBLASDX
                                              : mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<half_t, half_t, float>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x16_C32C16C16C32_TN_CUBLASDX
                                              : mma_atom::SM80_16x8x16_F32F16F16F32_TN;
                    } else if constexpr (SM >= 750) {
                        ret = is_complex_type ? mma_atom::SM75_16x8x8_C32C16C16C32_TN_CUBLASDX
                                              : mma_atom::SM75_16x8x8_F32F16F16F32_TN_CUBLASDX;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<double, double, double>>) {
                    if constexpr (SM > 900) {
                        // do not emulate Hopper DMMA on Blackwell+
                        ret = is_complex_type ? mma_atom::SM80_8x8x4_C64C64C64C64_TN
                                              : mma_atom::SM80_8x8x4_F64F64F64F64_TN;
                    } else if constexpr (SM == 900) {
                        ret = is_complex_type ? mma_atom::SM90_16x8x4_C64C64C64C64_TN
                                              : mma_atom::SM90_16x8x4_F64F64F64F64_TN;
                    } else if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_8x8x4_C64C64C64C64_TN
                                              : mma_atom::SM80_8x8x4_F64F64F64F64_TN;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<bfloat16_t, bfloat16_t, float>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x16_C32BC16BC16C32_TN_CUBLASDX
                                              : mma_atom::SM80_16x8x16_F32BF16BF16F32_TN;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<tfloat32_t, tfloat32_t, float>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX
                                              : mma_atom::SM80_16x8x8_F32TF32TF32F32_TN;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<int8_t, int8_t, int32_t>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX
                                              : mma_atom::SM80_16x8x32_S32S8S8S32_TN;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<int8_t, uint8_t, int32_t>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x32_CS32CS8CU8CS32_TN_CUBLASDX
                                              : mma_atom::SM80_16x8x32_S32S8U8S32_TN;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<uint8_t, int8_t, int32_t>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x32_CS32CU8CS8CS32_TN_CUBLASDX
                                              : mma_atom::SM80_16x8x32_S32U8S8S32_TN;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<uint8_t, uint8_t, int32_t>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x32_CS32CU8CU8CS32_TN_CUBLASDX
                                              : mma_atom::SM80_16x8x32_S32U8U8S32_TN;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<float_e4m3_t, float_e4m3_t, float>>) {
                    if constexpr (SM >= 890) {
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C32CE4M3CE4M3C32_TN_CUBLASDX
                                              : mma_atom::SM89_16x8x32_F32E4M3E4M3F32_TN_CUBLASDX;
                    }

                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<float_e4m3_t, float_e5m2_t, float>>) {
                    if constexpr (SM >= 890) {
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C32CE4M3CE5M2C32_TN_CUBLASDX
                                              : mma_atom::SM89_16x8x32_F32E4M3E5M2F32_TN_CUBLASDX;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<float_e5m2_t, float_e4m3_t, float>>) {
                    if constexpr (SM >= 890) {
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C32CE5M2CE4M3C32_TN_CUBLASDX
                                              : mma_atom::SM89_16x8x32_F32E5M2E4M3F32_TN_CUBLASDX;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<float_e5m2_t, float_e5m2_t, float>>) {
                    if constexpr (SM >= 890) {
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX
                                              : mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<float_e4m3_t, float_e4m3_t, cute::half_t>>) {
                    if constexpr (SM >= 890) {
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C16CE4M3CE4M3C16_TN_CUBLASDX
                                              : mma_atom::SM89_16x8x32_F16E4M3E4M3F16_TN_CUBLASDX;
                    }

                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<float_e4m3_t, float_e5m2_t, cute::half_t>>) {
                    if constexpr (SM >= 890) {
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C16CE4M3CE5M2C16_TN_CUBLASDX
                                              : mma_atom::SM89_16x8x32_F16E4M3E5M2F16_TN_CUBLASDX;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<float_e5m2_t, float_e4m3_t, cute::half_t>>) {
                    if constexpr (SM >= 890) {
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C16CE5M2CE4M3C16_TN_CUBLASDX
                                              : mma_atom::SM89_16x8x32_F16E5M2E4M3F16_TN_CUBLASDX;
                    }
                } else if constexpr (cute::is_same_v<cute::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                     cute::tuple<float_e5m2_t, float_e5m2_t, cute::half_t>>) {
                    if constexpr (SM >= 890) {
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C16CE5M2CE5M2C16_TN_CUBLASDX
                                              : mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX;
                    }
                }

                return ret;
            }

            // This function returns instance of CuTe type
            // can be used with decltype to get just the type
            template<typename AType, typename BType, typename CType, mma_atom MmaAtom>
            constexpr auto convert_mma_atom_to_cute() {
                if constexpr (MmaAtom == mma_atom::universal_fma) {
                    return UniversalFMA<AType, BType, CType> {};
                } else if constexpr (MmaAtom == mma_atom::SM100_2x1x1_F32F32F32F32) {
                    return cublasdx::detail::SM100_2x1x1_F32F32F32F32 {};
                } else if constexpr (MmaAtom == mma_atom::SM100_1x2x1_F32F32F32F32) {
                    return cublasdx::detail::SM100_1x2x1_F32F32F32F32 {};
                } else if constexpr (MmaAtom == mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX) {
                    return SM75_16x8x8_F16F16F16F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM75_16x8x8_C16C16C16C16_TN_CUBLASDX) {
                    return SM75_16x8x8_C16C16C16C16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM75_16x8x8_F32F16F16F32_TN_CUBLASDX) {
                    return SM75_16x8x8_F32F16F16F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM75_16x8x8_C32C16C16C32_TN_CUBLASDX) {
                    return SM75_16x8x8_C32C16C16C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_F16F16F16F16_TN) {
                    return cute::SM80_16x8x16_F16F16F16F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_C16C16C16C16_TN_CUBLASDX) {
                    return SM80_16x8x16_C16C16C16C16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_F32F16F16F32_TN) {
                    return cute::SM80_16x8x16_F32F16F16F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_C32C16C16C32_TN_CUBLASDX) {
                    return SM80_16x8x16_C32C16C16C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_8x8x4_F64F64F64F64_TN) {
                    return cute::SM80_8x8x4_F64F64F64F64_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_8x8x4_C64C64C64C64_TN) {
                    return cute::SM80_8x8x4_C64C64C64C64_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x8_F32BF16BF16F32_TN) {
                    return cute::SM80_16x8x8_F32BF16BF16F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x8_C32BC16BC16C32_TN_CUBLASDX) {
                    return SM80_16x8x8_C32BC16BC16C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_F32BF16BF16F32_TN) {
                    return cute::SM80_16x8x16_F32BF16BF16F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_C32BC16BC16C32_TN_CUBLASDX) {
                    return SM80_16x8x16_C32BC16BC16C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x4_F32TF32TF32F32_TN) {
                    return cute::SM80_16x8x4_F32TF32TF32F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX) {
                    return SM80_16x8x4_C32TC32TC32C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x8_F32TF32TF32F32_TN) {
                    return cute::SM80_16x8x8_F32TF32TF32F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX) {
                    return SM80_16x8x8_C32TC32TC32C32_TN {};
                }
                // Int MMAs
                else if constexpr (MmaAtom == mma_atom::SM80_8x8x16_S32S8S8S32_TN) {
                    return cute::SM80_8x8x16_S32S8S8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_8x8x16_S32S8U8S32_TN) {
                    return cute::SM80_8x8x16_S32S8U8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_8x8x16_S32U8S8S32_TN) {
                    return cute::SM80_8x8x16_S32U8S8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_8x8x16_S32U8U8S32_TN) {
                    return cute::SM80_8x8x16_S32U8U8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_S32S8S8S32_TN) {
                    return cute::SM80_16x8x16_S32S8S8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_S32S8U8S32_TN) {
                    return cute::SM80_16x8x16_S32S8U8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_S32U8S8S32_TN) {
                    return cute::SM80_16x8x16_S32U8S8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_S32U8U8S32_TN) {
                    return cute::SM80_16x8x16_S32U8U8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x32_S32S8S8S32_TN) {
                    return cute::SM80_16x8x32_S32S8S8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x32_S32S8U8S32_TN) {
                    return cute::SM80_16x8x32_S32S8U8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x32_S32U8S8S32_TN) {
                    return cute::SM80_16x8x32_S32U8S8S32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x32_S32U8U8S32_TN) {
                    return cute::SM80_16x8x32_S32U8U8S32_TN {};
                }
                // Complex Int MMAs
                else if constexpr (MmaAtom == mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX) {
                    return SM80_8x8x16_CS32CS8CS8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_8x8x16_CS32CS8CU8CS32_TN_CUBLASDX) {
                    return SM80_8x8x16_CS32CS8CU8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_8x8x16_CS32CU8CS8CS32_TN_CUBLASDX) {
                    return SM80_8x8x16_CS32CU8CS8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_8x8x16_CS32CU8CU8CS32_TN_CUBLASDX) {
                    return SM80_8x8x16_CS32CU8CU8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_CS32CS8CS8CS32_TN_CUBLASDX) {
                    return SM80_16x8x16_CS32CS8CS8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_CS32CS8CU8CS32_TN_CUBLASDX) {
                    return SM80_16x8x16_CS32CS8CU8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_CS32CU8CS8CS32_TN_CUBLASDX) {
                    return SM80_16x8x16_CS32CU8CS8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x16_CS32CU8CU8CS32_TN_CUBLASDX) {
                    return SM80_16x8x16_CS32CU8CU8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX) {
                    return SM80_16x8x32_CS32CS8CS8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x32_CS32CS8CU8CS32_TN_CUBLASDX) {
                    return SM80_16x8x32_CS32CS8CU8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x32_CS32CU8CS8CS32_TN_CUBLASDX) {
                    return SM80_16x8x32_CS32CU8CS8CS32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM80_16x8x32_CS32CU8CU8CS32_TN_CUBLASDX) {
                    return SM80_16x8x32_CS32CU8CU8CS32_TN {};
                }
                // FP8 MMAs
                else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F32E4M3E4M3F32_TN_CUBLASDX) {
                    return SM89_16x8x32_F32E4M3E4M3F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C32CE4M3CE4M3C32_TN_CUBLASDX) {
                    return SM89_16x8x32_C32CE4M3CE4M3C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F32E4M3E5M2F32_TN_CUBLASDX) {
                    return SM89_16x8x32_F32E4M3E5M2F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C32CE4M3CE5M2C32_TN_CUBLASDX) {
                    return SM89_16x8x32_C32CE4M3CE5M2C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F32E5M2E4M3F32_TN_CUBLASDX) {
                    return SM89_16x8x32_F32E5M2E4M3F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C32CE5M2CE4M3C32_TN_CUBLASDX) {
                    return SM89_16x8x32_C32CE5M2CE4M3C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX) {
                    return SM89_16x8x32_F32E5M2E5M2F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX) {
                    return SM89_16x8x32_C32CE5M2CE5M2C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F16E4M3E4M3F16_TN_CUBLASDX) {
                    return SM89_16x8x32_F16E4M3E4M3F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C16CE4M3CE4M3C16_TN_CUBLASDX) {
                    return SM89_16x8x32_C16CE4M3CE4M3C16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F16E4M3E5M2F16_TN_CUBLASDX) {
                    return SM89_16x8x32_F16E4M3E5M2F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C16CE4M3CE5M2C16_TN_CUBLASDX) {
                    return SM89_16x8x32_C16CE4M3CE5M2C16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F16E5M2E4M3F16_TN_CUBLASDX) {
                    return SM89_16x8x32_F16E5M2E4M3F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C16CE5M2CE4M3C16_TN_CUBLASDX) {
                    return SM89_16x8x32_C16CE5M2CE4M3C16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX) {
                    return SM89_16x8x32_F16E5M2E5M2F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C16CE5M2CE5M2C16_TN_CUBLASDX) {
                    return SM89_16x8x32_C16CE5M2CE5M2C16_TN {};
                }

                else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_F32E4M3E4M3F32_TN_CUBLASDX) {
                    return SM89_16x8x16_F32E4M3E4M3F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_C32CE4M3CE4M3C32_TN_CUBLASDX) {
                    return SM89_16x8x16_C32CE4M3CE4M3C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_F32E4M3E5M2F32_TN_CUBLASDX) {
                    return SM89_16x8x16_F32E4M3E5M2F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_C32CE4M3CE5M2C32_TN_CUBLASDX) {
                    return SM89_16x8x16_C32CE4M3CE5M2C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_F32E5M2E4M3F32_TN_CUBLASDX) {
                    return SM89_16x8x16_F32E5M2E4M3F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_C32CE5M2CE4M3C32_TN_CUBLASDX) {
                    return SM89_16x8x16_C32CE5M2CE4M3C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_F32E5M2E5M2F32_TN_CUBLASDX) {
                    return SM89_16x8x16_F32E5M2E5M2F32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_C32CE5M2CE5M2C32_TN_CUBLASDX) {
                    return SM89_16x8x16_C32CE5M2CE5M2C32_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_F16E4M3E4M3F16_TN_CUBLASDX) {
                    return SM89_16x8x16_F16E4M3E4M3F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_C16CE4M3CE4M3C16_TN_CUBLASDX) {
                    return SM89_16x8x16_C16CE4M3CE4M3C16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_F16E4M3E5M2F16_TN_CUBLASDX) {
                    return SM89_16x8x16_F16E4M3E5M2F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_C16CE4M3CE5M2C16_TN_CUBLASDX) {
                    return SM89_16x8x16_C16CE4M3CE5M2C16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_F16E5M2E4M3F16_TN_CUBLASDX) {
                    return SM89_16x8x16_F16E5M2E4M3F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_C16CE5M2CE4M3C16_TN_CUBLASDX) {
                    return SM89_16x8x16_C16CE5M2CE4M3C16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX) {
                    return SM89_16x8x16_F16E5M2E5M2F16_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM89_16x8x16_C16CE5M2CE5M2C16_TN_CUBLASDX) {
                    return SM89_16x8x16_C16CE5M2CE5M2C16_TN {};
                }

                // FP64 MMAs
                else if constexpr (MmaAtom == mma_atom::SM90_16x8x4_F64F64F64F64_TN) {
                    return cute::SM90_16x8x4_F64F64F64F64_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM90_16x8x8_F64F64F64F64_TN) {
                    return cute::SM90_16x8x8_F64F64F64F64_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM90_16x8x16_F64F64F64F64_TN) {
                    return cute::SM90_16x8x16_F64F64F64F64_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM90_16x8x4_C64C64C64C64_TN) {
                    return cute::SM90_16x8x4_C64C64C64C64_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM90_16x8x8_C64C64C64C64_TN) {
                    return cute::SM90_16x8x8_C64C64C64C64_TN {};
                } else if constexpr (MmaAtom == mma_atom::SM90_16x8x16_C64C64C64C64_TN) {
                    return cute::SM90_16x8x16_C64C64C64C64_TN {};
                }
            }
        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP
