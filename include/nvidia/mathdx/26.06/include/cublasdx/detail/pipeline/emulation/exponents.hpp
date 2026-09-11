// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_EXPONENTS_HPP
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_EXPONENTS_HPP

#include "cublasdx/detail/commondx_config.hpp"

#ifdef COMMONDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/cstdint>
#    include <cuda/std/type_traits>
#else
#    include <cstdint>
#    include <type_traits>
#endif

namespace cublasdx {
    namespace detail {

        union emulation_fp64_structure {
            double d;
            struct float64 {
                unsigned int mantissa_lo : 32;
                unsigned int mantissa_hi : 20;
                unsigned int exponent : 11;
                unsigned int sign : 1;
            } s;
        };

        static constexpr int emulation_fp64_bias = 1023;

        template<class T>
        CUBLASDX_HOST_DEVICE constexpr int emulation_slice_width() {
            if constexpr (COMMONDX_STL_NAMESPACE::is_signed_v<T>) {
                return 8 * sizeof(T) - 1;
            } else {
                return 8 * sizeof(T);
            }
        }

        template<class T>
        CUBLASDX_HOST_DEVICE constexpr int emulation_max_exponent();

        template<>
        CUBLASDX_HOST_DEVICE constexpr int emulation_max_exponent<COMMONDX_STL_NAMESPACE::uint8_t>() {
            return 8;
        }

        template<>
        CUBLASDX_HOST_DEVICE constexpr int emulation_max_exponent<COMMONDX_STL_NAMESPACE::int8_t>() {
            return 7;
        }

        CUBLASDX_HOST_DEVICE COMMONDX_STL_NAMESPACE::int32_t emulation_get_exponent(double const val) {
            emulation_fp64_structure em = {val};
            int em_exponent = (em.s.exponent + 1 - emulation_fp64_bias);
            if ((em.s.mantissa_hi & (63 << 14)) == (63 << 14)) {
                em_exponent++;
            }
            return em_exponent;
        }

        CUBLASDX_HOST_DEVICE void emulation_epilogue_ldexp(emulation_fp64_structure& em, int const exp) {
            static constexpr int exp_max = emulation_fp64_bias - 1;
            int previous_exp_biased = static_cast<int>(em.s.exponent);
            if (0 < previous_exp_biased && 0 < previous_exp_biased + exp &&
                previous_exp_biased + exp <= exp_max + emulation_fp64_bias) {
                em.s.exponent += exp;
                return;
            }
            em.d = ldexp(em.d, exp);
        }

        CUBLASDX_HOST_DEVICE COMMONDX_STL_NAMESPACE::int32_t emulation_max_to_exponent_shift(double const row_col_max) {
            static constexpr int scale_max_exponent = emulation_max_exponent<COMMONDX_STL_NAMESPACE::int8_t>();
            return scale_max_exponent - emulation_get_exponent(row_col_max);
        }

        template<typename DiagonalAccType, typename SliceValueType>
        CUBLASDX_HOST_DEVICE double emulation_nth_slice_to_fp64(COMMONDX_STL_NAMESPACE::int32_t const nth,
                                                                DiagonalAccType const nth_slice,
                                                                COMMONDX_STL_NAMESPACE::int32_t const exponent_shift) {
            static_assert(COMMONDX_STL_NAMESPACE::is_integral_v<DiagonalAccType>);
            static_assert(COMMONDX_STL_NAMESPACE::is_signed_v<DiagonalAccType>);
            static_assert(COMMONDX_STL_NAMESPACE::is_integral_v<SliceValueType>);
            static_assert(COMMONDX_STL_NAMESPACE::is_signed_v<SliceValueType>);

            constexpr int slice_width = emulation_slice_width<COMMONDX_STL_NAMESPACE::make_unsigned_t<SliceValueType>>();
            emulation_fp64_structure value = {static_cast<double>(nth_slice)};
            emulation_epilogue_ldexp(value, -slice_width * nth - exponent_shift);
            return value.d;
        }

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_EMULATION_EXPONENTS_HPP
