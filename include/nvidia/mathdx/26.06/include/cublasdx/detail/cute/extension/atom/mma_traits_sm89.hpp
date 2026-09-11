// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM89_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM89_HPP

#include <cute/config.hpp>

#include "cublasdx/detail/system_checks.hpp"

#include "commondx/traits/numeric_traits.hpp"

#include "cublasdx/detail/cute/extension/arch/mma_sm89.hpp"

namespace cublasdx {
    namespace detail {
        using SM89_FP8_16x8x16_A = Layout<Shape<Shape<cute::_4, cute::_8>, Shape<cute::_4, cute::_2>>,
                                          Stride<Stride<cute::_64, cute::_1>, Stride<cute::_16, cute::_8>>>;
        using SM89_FP8_16x8x16_B = Layout<Shape<Shape<cute::_4, cute::_8>, Shape<cute::_4>>,
                                          Stride<Stride<cute::_32, cute::_1>, Stride<cute::_8>>>;
        using SM89_FP8_16x8x16_C = Layout<Shape<Shape<cute::_4, cute::_8>, Shape<cute::_2, cute::_2>>,
                                          Stride<Stride<cute::_32, cute::_1>, Stride<cute::_16, cute::_8>>>;


        using SM89_FP8_16x8x32_A = Layout<Shape<Shape<cute::_4, cute::_8>, Shape<cute::_4, cute::_2, cute::_2>>,
                                          Stride<Stride<cute::_64, cute::_1>, Stride<cute::_16, cute::_8, cute::_256>>>;
        using SM89_FP8_16x8x32_B = Layout<Shape<Shape<cute::_4, cute::_8>, Shape<cute::_4, cute::_2>>,
                                          Stride<Stride<cute::_32, cute::_1>, Stride<cute::_8, cute::_128>>>;
        using SM89_FP8_16x8x32_C = Layout<Shape<Shape<cute::_4, cute::_8>, Shape<cute::_2, cute::_2>>,
                                          Stride<Stride<cute::_32, cute::_1>, Stride<cute::_16, cute::_8>>>;

        template<typename A, typename B>
        struct MMA_Traits_SM89_16x8x16_F32F8F8F32_TN {
            using ValTypeD = float;
            using ValTypeA = A;
            using ValTypeB = B;
            using ValTypeC = float;

            using Shape_MNK = Shape<cute::_16, cute::_8, cute::_16>;
            using ThrID     = Layout<cute::_32>;
            using ALayout   = SM89_FP8_16x8x16_A;
            using BLayout   = SM89_FP8_16x8x16_B;
            using CLayout   = SM89_FP8_16x8x16_C;
        };

        template<typename A, typename B>
        struct MMA_Traits_SM89_16x8x16_F16F8F8F16_TN {
            using ValTypeD = cute::half_t;
            using ValTypeA = A;
            using ValTypeB = B;
            using ValTypeC = cute::half_t;

            using Shape_MNK = Shape<cute::_16, cute::_8, cute::_16>;
            using ThrID     = Layout<cute::_32>;
            using ALayout   = SM89_FP8_16x8x16_A;
            using BLayout   = SM89_FP8_16x8x16_B;
            using CLayout   = SM89_FP8_16x8x16_C;
        };

        template<typename A, typename B>
        struct MMA_Traits_SM89_16x8x32_F32F8F8F32_TN {
            using ValTypeD = float;
            using ValTypeA = A;
            using ValTypeB = B;
            using ValTypeC = float;

            using Shape_MNK = Shape<cute::_16, cute::_8, cute::_32>;
            using ThrID     = Layout<cute::_32>;
            using ALayout   = SM89_FP8_16x8x32_A;
            using BLayout   = SM89_FP8_16x8x32_B;
            using CLayout   = SM89_FP8_16x8x32_C;
        };

        template<typename A, typename B>
        struct MMA_Traits_SM89_16x8x32_F16F8F8F16_TN {
            using ValTypeD = cute::half_t;
            using ValTypeA = A;
            using ValTypeB = B;
            using ValTypeC = cute::half_t;

            using Shape_MNK = Shape<cute::_16, cute::_8, cute::_32>;
            using ThrID     = Layout<cute::_32>;
            using ALayout   = SM89_FP8_16x8x32_A;
            using BLayout   = SM89_FP8_16x8x32_B;
            using CLayout   = SM89_FP8_16x8x32_C;
        };

    } // namespace detail
} // namespace cublasdx

namespace cute {

    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F32E4M3E4M3F32_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x32_F32F8F8F32_TN<cutlass::float_e4m3_t, cutlass::float_e4m3_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F32E4M3E5M2F32_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x32_F32F8F8F32_TN<cutlass::float_e4m3_t, cutlass::float_e5m2_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F32E5M2E4M3F32_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x32_F32F8F8F32_TN<cutlass::float_e5m2_t, cutlass::float_e4m3_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F32E5M2E5M2F32_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x32_F32F8F8F32_TN<cutlass::float_e5m2_t, cutlass::float_e5m2_t> {
    };

    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F16E4M3E4M3F16_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x32_F16F8F8F16_TN<cutlass::float_e4m3_t, cutlass::float_e4m3_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F16E4M3E5M2F16_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x32_F16F8F8F16_TN<cutlass::float_e4m3_t, cutlass::float_e5m2_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F16E5M2E4M3F16_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x32_F16F8F8F16_TN<cutlass::float_e5m2_t, cutlass::float_e4m3_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F16E5M2E5M2F16_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x32_F16F8F8F16_TN<cutlass::float_e5m2_t, cutlass::float_e5m2_t> {
    };

    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x16_F32E4M3E4M3F32_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x16_F32F8F8F32_TN<cutlass::float_e4m3_t, cutlass::float_e4m3_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x16_F32E4M3E5M2F32_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x16_F32F8F8F32_TN<cutlass::float_e4m3_t, cutlass::float_e5m2_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x16_F32E5M2E4M3F32_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x16_F32F8F8F32_TN<cutlass::float_e5m2_t, cutlass::float_e4m3_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x16_F32E5M2E5M2F32_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x16_F32F8F8F32_TN<cutlass::float_e5m2_t, cutlass::float_e5m2_t> {
    };

    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x16_F16E4M3E4M3F16_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x16_F16F8F8F16_TN<cutlass::float_e4m3_t, cutlass::float_e4m3_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x16_F16E4M3E5M2F16_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x16_F16F8F8F16_TN<cutlass::float_e4m3_t, cutlass::float_e5m2_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x16_F16E5M2E4M3F16_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x16_F16F8F8F16_TN<cutlass::float_e5m2_t, cutlass::float_e4m3_t> {
    };
    template<>
    struct MMA_Traits<cublasdx::detail::SM89_16x8x16_F16E5M2E5M2F16_TN>:
        public cublasdx::detail::MMA_Traits_SM89_16x8x16_F16F8F8F16_TN<cutlass::float_e5m2_t, cutlass::float_e5m2_t> {
    };

} // namespace cute

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_MMA_TRAITS_SM89_HPP
