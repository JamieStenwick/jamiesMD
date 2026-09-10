// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_HPP

#include <cute/config.hpp>

#include "cublasdx/detail/system_checks.hpp"

#include "commondx/traits/numeric_traits.hpp"

namespace cublasdx {
    namespace detail {
        using cute::Int;
        using cute::Layout;
        using cute::Shape;
        using cute::Stride;
        using cute::Tensor;

        template<class A, class B>
        constexpr auto get_multiplication_type_helper() {
            constexpr bool is_complex = cutlass::is_complex<A>::value and cutlass::is_complex<B>::value;
            constexpr bool is_real = not cutlass::is_complex<A>::value and not cutlass::is_complex<B>::value;
            static_assert(is_complex or is_real, "Invalid multiplication type, must be either complex or real");

            if constexpr(is_complex) {
                using value_type = decltype(get_multiplication_type_helper<typename A::value_type, typename B::value_type>());
                const auto tmp = static_cast<value_type>(0.f);
                return cutlass::complex<value_type>(tmp, tmp);
            } else {
                constexpr bool is_integral = commondx::is_integral_v<A> or commondx::is_integral_v<B>;
                constexpr bool is_floating = commondx::is_floating_point_v<A> or commondx::is_floating_point_v<B>;
                static_assert(is_integral or is_floating, "Invalid multiplication type, must be either integral or floating");

                if constexpr(is_integral) {
                    // This is a well defined operation in C++
                    return decltype(cute::declval<A>() * cute::declval<B>())(0);
                } else {
                    constexpr bool is_one_fp64 = cute::is_same_v<A, double> or cute::is_same_v<B, double>;
                    constexpr bool is_one_fp32 = cute::is_same_v<A, float> or cute::is_same_v<B, float>;
                    constexpr bool is_one_tf32 = cute::is_same_v<A, tfloat32_t> or cute::is_same_v<B, tfloat32_t>;
                    constexpr bool is_one_fp16 = cute::is_same_v<A, cute::half_t> or cute::is_same_v<B, cute::half_t>;
                    constexpr bool are_both_fp16 = cute::is_same_v<A, cute::half_t> and cute::is_same_v<B, cute::half_t>;
                    constexpr bool is_one_bf16 = cute::is_same_v<A, cute::bfloat16_t> or cute::is_same_v<B, cute::bfloat16_t>;
                    constexpr bool is_one_fp8_e5m2 = cute::is_same_v<A, float_e5m2_t> or cute::is_same_v<B, float_e5m2_t>;
                    constexpr bool are_both_fp8_e5m2 = cute::is_same_v<A, float_e5m2_t> and cute::is_same_v<B, float_e5m2_t>;
                    constexpr bool is_one_fp8_e4m3 = cute::is_same_v<A, float_e4m3_t> or cute::is_same_v<B, float_e4m3_t>;
                    constexpr bool are_both_fp8_e4m3 = cute::is_same_v<A, float_e4m3_t> and cute::is_same_v<B, float_e4m3_t>;
                    constexpr bool is_one_fp8 = is_one_fp8_e5m2 or is_one_fp8_e4m3; 
                    constexpr bool are_both_fp8 = are_both_fp8_e5m2 or are_both_fp8_e4m3 or (is_one_fp8_e5m2 and is_one_fp8_e4m3);

                    constexpr bool is_correct_type = is_one_fp64 or is_one_fp32 or is_one_tf32 or is_one_fp16 or are_both_fp16
                                                     or is_one_fp8 or is_one_bf16 or are_both_fp8;
                    
                    if constexpr(is_one_fp64) {
                        return static_cast<double>(0.0);
                    } else if constexpr(is_one_fp32 or is_one_tf32) {
                        return static_cast<float>(0.f);
                    } else if constexpr(are_both_fp16) {
                        return static_cast<cute::half_t>(0.f);
                    } else if constexpr(is_one_fp16 and is_one_fp8) {
                        return static_cast<cute::half_t>(0.f);
                    } else if constexpr(is_one_bf16) {
                        return static_cast<float>(0.f);
                    } else if constexpr(are_both_fp8) {
                        return static_cast<cute::half_t>(0.f);
                    } else if constexpr(is_one_fp8){
                        return static_cast<float>(0.f);
                    } else {
                        static_assert(is_correct_type, "Invalid multiplication type");
                    }
                }
            }
        }

        template<class A, class B>
        struct get_multiplication_type {
            using type = decltype(get_multiplication_type_helper<A, B>());
        };

        template<class A, class B>
        using get_multiplication_type_t = typename get_multiplication_type<A, B>::type;

        // General FMA
        template<class A, class B, class C>
        CUTE_HOST_DEVICE constexpr void fma(C& d, A const& a, B const& b, C const& c) {
            using multiplication_type = get_multiplication_type_t<A, B>;
            static_assert(not has_complex_interface_v<multiplication_type>, "Only cutlass::complex<T> is supported for FMA");
            if constexpr (cute::is_same_v<A, float> and cute::is_same_v<B, float> and cute::is_same_v<C, float>) {
#if defined(__CUDA_ARCH__)
                d = __fmaf_rn(a, b, c);
#else
                d = a * b + c;
#endif
            } else {
                d = static_cast<C>(static_cast<multiplication_type>(a) * static_cast<multiplication_type>(b)) + c;
            }
        }

        template<class A, class B, class C>
        CUTE_HOST_DEVICE constexpr void mul(C& d, A const& a, B const& b) {
            using multiplication_type = get_multiplication_type_t<A, B>;
            static_assert(not has_complex_interface_v<multiplication_type>, "Only cutlass::complex<T> is supported for MUL");
            d = static_cast<C>(static_cast<multiplication_type>(a) * static_cast<multiplication_type>(b));
        }

        // d can be aliased with c but not a or b
        template<class A, class B, class C>
        CUTE_HOST_DEVICE constexpr void fma(cutlass::complex<C>&       d,
                                            cutlass::complex<A> const& a,
                                            cutlass::complex<B> const& b,
                                            cutlass::complex<C> const& c) {
            fma(d.real(), a.real(), b.real(), c.real());
            fma(d.imag(), a.real(), b.imag(), c.imag());
            fma(d.real(), static_cast<A>(-a.imag()), b.imag(), d.real());
            fma(d.imag(), a.imag(), b.real(), d.imag());
        }

        // d can be aliased with a or b to allow for A = A * B style operations
        template<class A, class B, class C>
        CUTE_HOST_DEVICE constexpr void mul(cutlass::complex<C>& d, 
                                            cutlass::complex<A> const& a, 
                                            cutlass::complex<B> const& b) {
            cutlass::complex<C> tmp; // allow aliasing of d with a or b
            mul(tmp.real(), a.real(), b.real());
            mul(tmp.imag(), a.real(), b.imag());
            fma(tmp.real(), static_cast<A>(-a.imag()), b.imag(), tmp.real());
            fma(tmp.imag(), a.imag(), b.real(), tmp.imag());
            d = tmp;
        }
        
        // Universal FMA
        template<class A, class B, class C>
        struct UniversalFMA {
            using DRegisters = C[1];
            using ARegisters = A[1];
            using BRegisters = B[1];
            using CRegisters = C[1];

            CUTE_HOST_DEVICE static constexpr void fma(C& d, A const& a, B const& b, C const& c) {
                using cublasdx::detail::fma;
                fma(d, a, b, c);
            }
        };

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_MMA_HPP
