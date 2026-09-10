// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_PRECISION_HPP
#define CUBLASDX_OPERATORS_PRECISION_HPP

#include "commondx/operators/precision.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cublasdx/types.hpp"

namespace cublasdx {


    template<class T>
#if COMMONDX_DETAIL_CUDA_FP8_ENABLED
    struct PrecisionCheck:
        public commondx::PrecisionBase<T,
                                       __nv_fp8_e5m2,
                                       __nv_fp8_e4m3,
                                       __half,
                                       __nv_bfloat16,
                                       tfloat32_t,
                                       float,
                                       double, // floating point
                                       int8_t,
                                       uint8_t,
                                       int16_t,
                                       uint16_t,
                                       int32_t,
                                       uint32_t,
                                       int64_t,
                                       uint64_t> { // integral
#else
    struct PrecisionCheck:
        public commondx::PrecisionBase<T,
                                       __half,
                                       __nv_bfloat16,
                                       tfloat32_t,
                                       float,
                                       double, // floating point
                                       int8_t,
                                       uint8_t,
                                       int16_t,
                                       uint16_t,
                                       int32_t,
                                       uint32_t,
                                       int64_t,
                                       uint64_t> { // integral
#endif
    };

    template<class PA, class PB = PA, class PC = PA>
    struct Precision: public commondx::detail::operator_expression {
        using a_type = typename PrecisionCheck<PA>::type;
        using b_type = typename PrecisionCheck<PB>::type;
        using c_type = typename PrecisionCheck<PC>::type;
    };

    namespace detail {
        using default_blas_precision_operator = Precision<float, float, float>;
    } // namespace detail
} // namespace cublasdx

namespace commondx::detail {
    template<class PA, class PB, class PC>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::precision, cublasdx::Precision<PA, PB, PC>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<class PA, class PB, class PC>
    struct get_operator_type<cublasdx::operator_type, cublasdx::Precision<PA, PB, PC>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::precision;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_PRECISION_HPP
