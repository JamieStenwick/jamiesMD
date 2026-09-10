// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_DATABASE_REDIRECT_HPP
#define CUBLASDX_DATABASE_DATABASE_REDIRECT_HPP

#include "cublasdx/detail/cute/cute_tensor.hpp"
#include "cublasdx/database/gemm/mma_atoms.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {
            // Some MMAs mirror each other (such as FP32FP16FP16FP32 and FP32BF16BF16FP32)
            // and they are all redirected to one matching config in the database
            template<typename TA, typename TB, typename TC, typename SM, typename EnableIfHelper = void>
            struct database_precision_redirect {
                using a_type = TA;
                using b_type = TB;
                using c_type = TC;

                static constexpr bool decayed = false;
            };

            // Use FP16 config for FP32BF16 compute
            template<typename SM>
            struct database_precision_redirect<bfloat16_t, bfloat16_t, float, SM, cute::enable_if_t<SM::value >= 800>> {
                using a_type = half_t;
                using b_type = half_t;
                using c_type = float;

                static constexpr bool decayed = true;
            };

            // Use INT8 config for UINT8 compute
            template<typename TA, typename TB, typename SM>
            struct database_precision_redirect<
                TA,
                TB,
                int32_t,
                SM,
                cute::enable_if_t<cute::is_same_v<TA, uint8_t> or cute::is_same_v<TB, uint8_t>>> {
                using a_type = cute::conditional_t<cute::is_same_v<TA, uint8_t>, int8_t, TA>;
                using b_type = cute::conditional_t<cute::is_same_v<TB, uint8_t>, int8_t, TB>;
                using c_type = int32_t;

                static constexpr bool decayed = true;
            };

            // Use E5M2 config for E4M3 compute FP32
            template<typename TA, typename TB, typename SM>
            struct database_precision_redirect<
                TA,
                TB,
                float,
                SM,
                cute::enable_if_t<cute::is_same_v<TA, float_e4m3_t> or cute::is_same_v<TB, float_e4m3_t>>> {
                using a_type = cute::conditional_t<cute::is_same_v<TA, float_e4m3_t>, float_e5m2_t, TA>;
                using b_type = cute::conditional_t<cute::is_same_v<TB, float_e4m3_t>, float_e5m2_t, TB>;
                using c_type = float;

                static constexpr bool decayed = true;
            };

            // Use E5M2 config for E4M3 compute FP16
            template<typename TA, typename TB, typename SM>
            struct database_precision_redirect<
                TA,
                TB,
                cute::half_t,
                SM,
                cute::enable_if_t<cute::is_same_v<TA, float_e4m3_t> or cute::is_same_v<TB, float_e4m3_t>>> {
                using a_type = cute::conditional_t<cute::is_same_v<TA, float_e4m3_t>, float_e5m2_t, TA>;
                using b_type = cute::conditional_t<cute::is_same_v<TB, float_e4m3_t>, float_e5m2_t, TB>;
                using c_type = cute::half_t;

                static constexpr bool decayed = true;
            };

            template<typename... Ts>
            using database_precision_redirect_a_t = typename database_precision_redirect<Ts...>::a_type;

            template<typename... Ts>
            using database_precision_redirect_b_t = typename database_precision_redirect<Ts...>::b_type;

            template<typename... Ts>
            using database_precision_redirect_c_t = typename database_precision_redirect<Ts...>::c_type;

            template<typename... Ts>
            constexpr bool database_precision_redirect_v = database_precision_redirect<Ts...>::decayed;

            template<typename TA, typename TB, typename TC, typename SM>
            struct database_precision_redirect<
                TA,
                TB,
                TC,
                SM,
                cute::enable_if_t<cutlass::is_complex<TA>::value && cutlass::is_complex<TB>::value &&
                                  cutlass::is_complex<TC>::value>> {

                using a_value_t = typename TA::value_type;
                using b_value_t = typename TB::value_type;
                using c_value_t = typename TC::value_type;

                using value_type_decay = database_precision_redirect<a_value_t, b_value_t, c_value_t, SM>;

                using a_type = cutlass::complex<typename value_type_decay::a_type>;
                using b_type = cutlass::complex<typename value_type_decay::b_type>;
                using c_type = cutlass::complex<typename value_type_decay::c_type>;

                static constexpr bool decayed = value_type_decay::decayed;
            };

            // This should be a macro, it's a partial function used to change AA into
            // AB, BA, BB permutation. To be changed into a cleaner non-partial solution
            template<class TA, class TB, class CA, class CB>
            constexpr mma_atom combination_dispatch(mma_atom aa, mma_atom bb, mma_atom ba, mma_atom ab) {
                if constexpr (cute::is_same_v<TA, CB> && cute::is_same_v<TB, CB>) {
                    return bb;
                } else if constexpr (cute::is_same_v<TA, CB> && cute::is_same_v<TB, CA>) {
                    return ba;
                } else if constexpr (cute::is_same_v<TA, CA> && cute::is_same_v<TB, CB>) {
                    return ab;
                }

                return aa;
            }

#define COMBINATION_DISPATCH_FP8_MMA_REAL(INSTRUCTION_SIZE, ACCUMULATOR_BITS, TA, TB)                           \
    case mma_atom::SM89_##INSTRUCTION_SIZE##_F##ACCUMULATOR_BITS##E5M2E5M2F##ACCUMULATOR_BITS##_TN_CUBLASDX:    \
        return combination_dispatch<TA, TB, float_e5m2_t, float_e4m3_t>(                                        \
            mma_atom::SM89_##INSTRUCTION_SIZE##_F##ACCUMULATOR_BITS##E5M2E5M2F##ACCUMULATOR_BITS##_TN_CUBLASDX, \
            mma_atom::SM89_##INSTRUCTION_SIZE##_F##ACCUMULATOR_BITS##E4M3E4M3F##ACCUMULATOR_BITS##_TN_CUBLASDX, \
            mma_atom::SM89_##INSTRUCTION_SIZE##_F##ACCUMULATOR_BITS##E4M3E5M2F##ACCUMULATOR_BITS##_TN_CUBLASDX, \
            mma_atom::SM89_##INSTRUCTION_SIZE##_F##ACCUMULATOR_BITS##E5M2E4M3F##ACCUMULATOR_BITS##_TN_CUBLASDX);

#define COMBINATION_DISPATCH_FP8_MMA_COMPLEX(INSTRUCTION_SIZE, ACCUMULATOR_BITS, TA, TB)                          \
    case mma_atom::SM89_##INSTRUCTION_SIZE##_C##ACCUMULATOR_BITS##CE5M2CE5M2C##ACCUMULATOR_BITS##_TN_CUBLASDX:    \
        return combination_dispatch<TA, TB, cutlass::complex<float_e5m2_t>, cutlass::complex<float_e4m3_t>>(      \
            mma_atom::SM89_##INSTRUCTION_SIZE##_C##ACCUMULATOR_BITS##CE5M2CE5M2C##ACCUMULATOR_BITS##_TN_CUBLASDX, \
            mma_atom::SM89_##INSTRUCTION_SIZE##_C##ACCUMULATOR_BITS##CE4M3CE4M3C##ACCUMULATOR_BITS##_TN_CUBLASDX, \
            mma_atom::SM89_##INSTRUCTION_SIZE##_C##ACCUMULATOR_BITS##CE4M3CE5M2C##ACCUMULATOR_BITS##_TN_CUBLASDX, \
            mma_atom::SM89_##INSTRUCTION_SIZE##_C##ACCUMULATOR_BITS##CE5M2CE4M3C##ACCUMULATOR_BITS##_TN_CUBLASDX);

#define COMBINATION_DISPATCH_INT8_MMA_REAL(INSTRUCTION_SIZE, TA, TB)                                            \
    case mma_atom::SM80_##INSTRUCTION_SIZE##_S32S8S8S32_TN:                                                     \
        return combination_dispatch<TA, TB, int8_t, uint8_t>(mma_atom::SM80_##INSTRUCTION_SIZE##_S32S8S8S32_TN, \
                                                             mma_atom::SM80_##INSTRUCTION_SIZE##_S32U8U8S32_TN, \
                                                             mma_atom::SM80_##INSTRUCTION_SIZE##_S32U8S8S32_TN, \
                                                             mma_atom::SM80_##INSTRUCTION_SIZE##_S32S8U8S32_TN);

#define COMBINATION_DISPATCH_INT8_MMA_COMPLEX(INSTRUCTION_SIZE, TA, TB)                           \
    case mma_atom::SM80_##INSTRUCTION_SIZE##_CS32CS8CS8CS32_TN_CUBLASDX:                          \
        return combination_dispatch<TA, TB, cutlass::complex<int8_t>, cutlass::complex<uint8_t>>( \
            mma_atom::SM80_##INSTRUCTION_SIZE##_CS32CS8CS8CS32_TN_CUBLASDX,                       \
            mma_atom::SM80_##INSTRUCTION_SIZE##_CS32CU8CU8CS32_TN_CUBLASDX,                       \
            mma_atom::SM80_##INSTRUCTION_SIZE##_CS32CU8CS8CS32_TN_CUBLASDX,                       \
            mma_atom::SM80_##INSTRUCTION_SIZE##_CS32CS8CU8CS32_TN_CUBLASDX);

            // Since the database for e.g. BF16 is checked with FP16 the atom
            // needs to be switched afterwards to a matching BF16 one
            template<typename AType, typename BType, typename CType, typename SM>
            constexpr mma_atom map_mma_from_database(mma_atom atom) {
                if constexpr (database_precision_redirect_v<AType, BType, CType, SM>) {
                    switch (atom) {
                        // 16 bit types
                        case mma_atom::SM75_16x8x8_F32F16F16F32_TN_CUBLASDX:
                            return mma_atom::SM80_16x8x8_F32BF16BF16F32_TN;
                        case mma_atom::SM75_16x8x8_C32C16C16C32_TN_CUBLASDX:
                            return mma_atom::SM80_16x8x8_C32BC16BC16C32_TN_CUBLASDX;
                        case mma_atom::SM80_16x8x16_F32F16F16F32_TN: return mma_atom::SM80_16x8x16_F32BF16BF16F32_TN;
                        case mma_atom::SM80_16x8x16_C32C16C16C32_TN_CUBLASDX:
                            return mma_atom::SM80_16x8x16_C32BC16BC16C32_TN_CUBLASDX;

                            // 8 bit floating point types
                            COMBINATION_DISPATCH_FP8_MMA_REAL(16x8x16, 32, AType, BType);
                            COMBINATION_DISPATCH_FP8_MMA_COMPLEX(16x8x16, 32, AType, BType);
                            COMBINATION_DISPATCH_FP8_MMA_REAL(16x8x32, 32, AType, BType);
                            COMBINATION_DISPATCH_FP8_MMA_COMPLEX(16x8x32, 32, AType, BType);
                            COMBINATION_DISPATCH_FP8_MMA_REAL(16x8x16, 16, AType, BType);
                            COMBINATION_DISPATCH_FP8_MMA_COMPLEX(16x8x16, 16, AType, BType);
                            COMBINATION_DISPATCH_FP8_MMA_REAL(16x8x32, 16, AType, BType);
                            COMBINATION_DISPATCH_FP8_MMA_COMPLEX(16x8x32, 16, AType, BType);

                            // 8 bit integer types
                            COMBINATION_DISPATCH_INT8_MMA_REAL(8x8x16, AType, BType);
                            COMBINATION_DISPATCH_INT8_MMA_COMPLEX(8x8x16, AType, BType);
                            COMBINATION_DISPATCH_INT8_MMA_REAL(16x8x16, AType, BType);
                            COMBINATION_DISPATCH_INT8_MMA_COMPLEX(16x8x16, AType, BType);
                            COMBINATION_DISPATCH_INT8_MMA_REAL(16x8x32, AType, BType);
                            COMBINATION_DISPATCH_INT8_MMA_COMPLEX(16x8x32, AType, BType);

                        default: {
                            return atom;
                        }
                    }
                }

                return atom;
            }
        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx
#endif // CUBLASDX_DATABASE_DATABASE_REDIRECT_HPP
