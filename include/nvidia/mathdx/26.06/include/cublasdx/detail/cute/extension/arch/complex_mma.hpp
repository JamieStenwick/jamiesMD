// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_COMPLEX_MMA_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_COMPLEX_MMA_HPP

#include <cute/config.hpp>

#include "cublasdx/detail/system_checks.hpp"

#include "commondx/traits/numeric_traits.hpp"

#include "cublasdx/detail/cute/extension/atom/mma_traits_sm75.hpp"
#include "cublasdx/detail/cute/extension/atom/mma_traits_sm89.hpp"

namespace cublasdx {
    namespace detail {
        using cute::Int;
        using cute::Layout;
        using cute::Shape;
        using cute::Stride;
        using cute::Tensor;
        // Complex MMAs
        template<class T, class Layout, class X>
        CUTE_HOST_DEVICE void riri_to_rrii(Tensor<T, Layout> const& complex_in, X* real_out, X* imag_out) {
            CUTE_UNROLL
            for (unsigned i = 0; i < size(complex_in); i++) {
                real_out[i] = complex_in[i].real();
                imag_out[i] = complex_in[i].imag();
            }
        }

        template<class T, class Layout, class X>
        CUTE_HOST_DEVICE void rrii_to_riri(Tensor<T, Layout>& complex_out, X* const real_in, X* const imag_in) {
            CUTE_UNROLL
            for (unsigned i = 0; i < size(complex_out); i++) {
                complex_out[i].real(real_in[i]);
                complex_out[i].imag(imag_in[i]);
            }
        }

        template<class X>
        CUTE_HOST_DEVICE void negate(X* a, unsigned nelem) {
            CUTE_UNROLL
            for (unsigned i = 0; i < nelem; i++) {
                a[i] = -a[i];
            }
        }

        using cute::MMA_Traits;
        template<class RealMMA,
                 typename d_value_type = typename MMA_Traits<RealMMA>::ValTypeD,
                 typename a_value_type = typename MMA_Traits<RealMMA>::ValTypeA,
                 typename b_value_type = typename MMA_Traits<RealMMA>::ValTypeB,
                 typename c_value_type = typename MMA_Traits<RealMMA>::ValTypeC>

        struct ComplexMMA {
            using DRegisters =
                complex<d_value_type>[(2 * sizeof(typename RealMMA::DRegisters)) /
                                      sizeof(complex<d_value_type>)]; // Number of registers are picked so we have
            // 2 (one for real and one for imaginary) x sizeof(RealMMA::A/B/C/D Registers) = sizeof(ComplexMMA::A/B/C/D Registers)
            using ARegisters =
                complex<a_value_type>[(2 * sizeof(typename RealMMA::ARegisters)) / sizeof(complex<a_value_type>)];
            using BRegisters =
                complex<b_value_type>[(2 * sizeof(typename RealMMA::BRegisters)) / sizeof(complex<b_value_type>)];
            using CRegisters =
                complex<c_value_type>[(2 * sizeof(typename RealMMA::CRegisters)) / sizeof(complex<c_value_type>)];

            template<class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
            CUTE_HOST_DEVICE static void fma(Tensor<TD, DLayout>&       rD,
                                             Tensor<TA, ALayout> const& rA,
                                             Tensor<TB, BLayout> const& rB,
                                             Tensor<TC, CLayout> const& rC) {
                typename RealMMA::DRegisters real_d, imag_d;
                typename RealMMA::ARegisters real_a, imag_a;
                typename RealMMA::BRegisters real_b, imag_b;
                typename RealMMA::CRegisters real_c, imag_c;

                constexpr int RealMMARegNumD = cute::extent<typename RealMMA::DRegisters>::value;
                constexpr int RealMMARegNumA = cute::extent<typename RealMMA::ARegisters>::value;
                constexpr int RealMMARegNumB = cute::extent<typename RealMMA::BRegisters>::value;
                constexpr int RealMMARegNumC = cute::extent<typename RealMMA::CRegisters>::value;

                riri_to_rrii(rA, reinterpret_cast<a_value_type*>(real_a), reinterpret_cast<a_value_type*>(imag_a));
                riri_to_rrii(rB, reinterpret_cast<b_value_type*>(real_b), reinterpret_cast<b_value_type*>(imag_b));
                riri_to_rrii(rC, reinterpret_cast<c_value_type*>(real_c), reinterpret_cast<c_value_type*>(imag_c));

                // d.real() =  a.real() * b.real() + c.real();
                cute::detail::explode(RealMMA::fma,
                                      real_d,
                                      cute::make_int_sequence<RealMMARegNumD> {},
                                      real_a,
                                      cute::make_int_sequence<RealMMARegNumA> {},
                                      real_b,
                                      cute::make_int_sequence<RealMMARegNumB> {},
                                      real_c,
                                      cute::make_int_sequence<RealMMARegNumC> {});

                // d.imag() =  a.imag() * b.real() + c.imag();
                cute::detail::explode(RealMMA::fma,
                                      imag_d,
                                      cute::make_int_sequence<RealMMARegNumD> {},
                                      imag_a,
                                      cute::make_int_sequence<RealMMARegNumA> {},
                                      real_b,
                                      cute::make_int_sequence<RealMMARegNumB> {},
                                      imag_c,
                                      cute::make_int_sequence<RealMMARegNumC> {});

                // d.real() = -a.imag() * b.imag() + d.real();
                negate(reinterpret_cast<a_value_type*>(imag_a),
                       sizeof(typename RealMMA::ARegisters) / sizeof(a_value_type));
                cute::detail::explode(RealMMA::fma,
                                      real_d,
                                      cute::make_int_sequence<RealMMARegNumD> {},
                                      imag_a,
                                      cute::make_int_sequence<RealMMARegNumA> {},
                                      imag_b,
                                      cute::make_int_sequence<RealMMARegNumB> {},
                                      real_d,
                                      cute::make_int_sequence<RealMMARegNumD> {});

                // d.imag() =  a.real() * b.imag() + d.imag();
                cute::detail::explode(RealMMA::fma,
                                      imag_d,
                                      cute::make_int_sequence<RealMMARegNumD> {},
                                      real_a,
                                      cute::make_int_sequence<RealMMARegNumA> {},
                                      imag_b,
                                      cute::make_int_sequence<RealMMARegNumB> {},
                                      imag_d,
                                      cute::make_int_sequence<RealMMARegNumD> {});

                rrii_to_riri(rD, reinterpret_cast<d_value_type*>(real_d), reinterpret_cast<d_value_type*>(imag_d));
            }
        };

        // FP16
        using SM75_16x8x8_C16C16C16C16_TN  = ComplexMMA<SM75_16x8x8_F16F16F16F16_TN>;
        using SM80_16x8x16_C16C16C16C16_TN = ComplexMMA<cute::SM80_16x8x16_F16F16F16F16_TN>;
        using SM75_16x8x8_C32C16C16C32_TN  = ComplexMMA<SM75_16x8x8_F32F16F16F32_TN>;

        // BF16
        using SM80_16x8x16_C32C16C16C32_TN   = ComplexMMA<cute::SM80_16x8x16_F32F16F16F32_TN>;
        using SM80_16x8x8_C32BC16BC16C32_TN  = ComplexMMA<cute::SM80_16x8x8_F32BF16BF16F32_TN>;
        using SM80_16x8x16_C32BC16BC16C32_TN = ComplexMMA<cute::SM80_16x8x16_F32BF16BF16F32_TN>;

        // TF32
        using SM80_16x8x4_C32TC32TC32C32_TN = ComplexMMA<cute::SM80_16x8x4_F32TF32TF32F32_TN>;
        using SM80_16x8x8_C32TC32TC32C32_TN = ComplexMMA<cute::SM80_16x8x8_F32TF32TF32F32_TN>;

        // FP64
        using SM80_8x8x4_C64C64C64C64_TN = ComplexMMA<cute::SM80_8x8x4_F64F64F64F64_TN>;

        // FP8
        using SM89_16x8x16_C32CE4M3CE4M3C32_TN = ComplexMMA<SM89_16x8x16_F32E4M3E4M3F32_TN>;
        using SM89_16x8x16_C32CE4M3CE5M2C32_TN = ComplexMMA<SM89_16x8x16_F32E4M3E5M2F32_TN>;
        using SM89_16x8x16_C32CE5M2CE4M3C32_TN = ComplexMMA<SM89_16x8x16_F32E5M2E4M3F32_TN>;
        using SM89_16x8x16_C32CE5M2CE5M2C32_TN = ComplexMMA<SM89_16x8x16_F32E5M2E5M2F32_TN>;
        using SM89_16x8x16_C16CE4M3CE4M3C16_TN = ComplexMMA<SM89_16x8x16_F16E4M3E4M3F16_TN>;
        using SM89_16x8x16_C16CE4M3CE5M2C16_TN = ComplexMMA<SM89_16x8x16_F16E4M3E5M2F16_TN>;
        using SM89_16x8x16_C16CE5M2CE4M3C16_TN = ComplexMMA<SM89_16x8x16_F16E5M2E4M3F16_TN>;
        using SM89_16x8x16_C16CE5M2CE5M2C16_TN = ComplexMMA<SM89_16x8x16_F16E5M2E5M2F16_TN>;

        using SM89_16x8x32_C32CE4M3CE4M3C32_TN = ComplexMMA<SM89_16x8x32_F32E4M3E4M3F32_TN>;
        using SM89_16x8x32_C32CE4M3CE5M2C32_TN = ComplexMMA<SM89_16x8x32_F32E4M3E5M2F32_TN>;
        using SM89_16x8x32_C32CE5M2CE4M3C32_TN = ComplexMMA<SM89_16x8x32_F32E5M2E4M3F32_TN>;
        using SM89_16x8x32_C32CE5M2CE5M2C32_TN = ComplexMMA<SM89_16x8x32_F32E5M2E5M2F32_TN>;
        using SM89_16x8x32_C16CE4M3CE4M3C16_TN = ComplexMMA<SM89_16x8x32_F16E4M3E4M3F16_TN>;
        using SM89_16x8x32_C16CE4M3CE5M2C16_TN = ComplexMMA<SM89_16x8x32_F16E4M3E5M2F16_TN>;
        using SM89_16x8x32_C16CE5M2CE4M3C16_TN = ComplexMMA<SM89_16x8x32_F16E5M2E4M3F16_TN>;
        using SM89_16x8x32_C16CE5M2CE5M2C16_TN = ComplexMMA<SM89_16x8x32_F16E5M2E5M2F16_TN>;

        // Int MMAs
        using SM80_8x8x16_CS32CS8CS8CS32_TN  = ComplexMMA<cute::SM80_8x8x16_S32S8S8S32_TN>;
        using SM80_8x8x16_CS32CS8CU8CS32_TN  = ComplexMMA<cute::SM80_8x8x16_S32S8U8S32_TN>;
        using SM80_8x8x16_CS32CU8CS8CS32_TN  = ComplexMMA<cute::SM80_8x8x16_S32U8S8S32_TN>;
        using SM80_8x8x16_CS32CU8CU8CS32_TN  = ComplexMMA<cute::SM80_8x8x16_S32U8U8S32_TN>;
        using SM80_16x8x16_CS32CS8CS8CS32_TN = ComplexMMA<cute::SM80_16x8x16_S32S8S8S32_TN>;
        using SM80_16x8x16_CS32CS8CU8CS32_TN = ComplexMMA<cute::SM80_16x8x16_S32S8U8S32_TN>;
        using SM80_16x8x16_CS32CU8CS8CS32_TN = ComplexMMA<cute::SM80_16x8x16_S32U8S8S32_TN>;
        using SM80_16x8x16_CS32CU8CU8CS32_TN = ComplexMMA<cute::SM80_16x8x16_S32U8U8S32_TN>;
        using SM80_16x8x32_CS32CS8CS8CS32_TN = ComplexMMA<cute::SM80_16x8x32_S32S8S8S32_TN>;
        using SM80_16x8x32_CS32CS8CU8CS32_TN = ComplexMMA<cute::SM80_16x8x32_S32S8U8S32_TN>;
        using SM80_16x8x32_CS32CU8CS8CS32_TN = ComplexMMA<cute::SM80_16x8x32_S32U8S8S32_TN>;
        using SM80_16x8x32_CS32CU8CU8CS32_TN = ComplexMMA<cute::SM80_16x8x32_S32U8U8S32_TN>;
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_COMPLEX_MMA_HPP
