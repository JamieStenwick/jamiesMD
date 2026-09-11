// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_AXPBY_HPP
#define CUBLASDX_DETAIL_AXPBY_HPP

#include "cublasdx/detail/blas_backend.hpp"
#include "cublasdx/detail/functional.hpp"
#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/detail/cute/extension/arch/mma.hpp"

namespace cublasdx {
    namespace detail {
        template<class Beta>
        CUBLASDX_HOST_DEVICE constexpr bool is_zero(Beta beta) {
            if constexpr (cutlass::is_complex<Beta>::value) {
                using vt        = typename Beta::value_type;
                const auto zero = static_cast<vt>(0.f);
                return beta.real() == zero && beta.imag() == zero;
            } else {
                const auto zero = static_cast<Beta>(0.f);
                return beta == zero;
            }
            CUTE_GCC_UNREACHABLE;
        }

        template<class Alpha, class XEngine, class XLayout, class Beta, class YEngine, class YLayout>
        CUBLASDX_HOST_DEVICE void axpby_impl(Alpha const&                          alpha,
                                             cute::Tensor<XEngine, XLayout> const& x_tensor,
                                             Beta const&                           beta,
                                             cute::Tensor<YEngine, YLayout>&       y_tensor) {
            using x_value_type = typename XEngine::value_type;
            using y_value_type = typename YEngine::value_type;

            using cutlass_x_value_type = detail::convert_to_cutlass_type_t<x_value_type>;
            using cutlass_y_value_type = detail::convert_to_cutlass_type_t<y_value_type>;

            auto cutlass_x_tensor = safe_recast<cutlass_x_value_type>(x_tensor);
            auto cutlass_y_tensor = safe_recast<cutlass_y_value_type>(y_tensor);

            if (is_zero(beta)) {
                CUTE_UNROLL
                for (int i = 0; i < cute::size(cutlass_y_tensor); ++i) {
                    cublasdx::detail::mul(cutlass_y_tensor(i), alpha, cutlass_x_tensor(i));
                }
            } else {
                CUTE_UNROLL
                for (int i = 0; i < cute::size(cutlass_y_tensor); ++i) {
                    cublasdx::detail::mul(cutlass_y_tensor(i), beta, cutlass_y_tensor(i));
                    cublasdx::detail::fma(cutlass_y_tensor(i), alpha, cutlass_x_tensor(i), cutlass_y_tensor(i));
                }
            }
        }

        template<class Alpha, class XEngine, class XLayout, class Beta, class YEngine, class YLayout>
        CUBLASDX_HOST_DEVICE void axpby_impl(Alpha const&                          alpha,
                                             cute::Tensor<XEngine, XLayout> const& x_tensor,
                                             Beta const&                           beta,
                                             cute::Tensor<YEngine, YLayout>&&      y_tensor) {
            axpby_impl(alpha, x_tensor, beta, y_tensor);
        }
    } // namespace detail

    template<class Alpha, class XEngine, class XLayout, class Beta, class YEngine, class YLayout>
    CUBLASDX_HOST_DEVICE void axpby(Alpha const&                          alpha,
                                    cute::Tensor<XEngine, XLayout> const& x_tensor,
                                    Beta const&                           beta,
                                    cute::Tensor<YEngine, YLayout>&       y_tensor) {

        using x_value_type = typename XEngine::value_type;
        using y_value_type = typename YEngine::value_type;

        static_assert(sizeof(Alpha) == sizeof(x_value_type) and alignof(Alpha) == alignof(x_value_type));
        static_assert(sizeof(Beta) == sizeof(y_value_type) and alignof(Beta) == alignof(y_value_type));

        detail::axpby_impl(reinterpret_cast<detail::convert_to_cutlass_type_t<x_value_type> const&>(alpha),
                           safe_recast<detail::convert_to_cutlass_type_t<x_value_type>>(x_tensor),
                           reinterpret_cast<detail::convert_to_cutlass_type_t<y_value_type> const&>(beta),
                           safe_recast<detail::convert_to_cutlass_type_t<y_value_type>>(y_tensor));
    }

    template<class Alpha, class XEngine, class XLayout, class Beta, class YEngine, class YLayout>
    CUBLASDX_HOST_DEVICE void axpby(Alpha const&                          alpha,
                                    cute::Tensor<XEngine, XLayout> const& x_tensor,
                                    Beta const&                           beta,
                                    cute::Tensor<YEngine, YLayout>&&      y_tensor) {
        cublasdx::axpby(alpha, x_tensor, beta, y_tensor);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_AXPBY_HPP
