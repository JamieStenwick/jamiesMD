// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_ENUM_HPP
#define CUSOLVERDX_DETAIL_ENUM_HPP

#include <cuComplex.h>

namespace cusolverdx::detail {
    enum class type_enum
    {
        real_f32,
        real_f64,
        complex_f32,
        complex_f64
    };

    constexpr __host__ __device__ bool type_enum_is_real(type_enum T) {
        return T == type_enum::real_f32 || T == type_enum::real_f64;
    }

    template<typename T>
    struct type_enum_of {
        static_assert(sizeof(T) == 0, "No type_enum mapping for T; supported types are float, double, cuFloatComplex, cuDoubleComplex");
    };
    template<> struct type_enum_of<float>                      { static constexpr type_enum value = type_enum::real_f32; };
    template<> struct type_enum_of<double>                     { static constexpr type_enum value = type_enum::real_f64; };
    template<> struct type_enum_of<cuFloatComplex>             { static constexpr type_enum value = type_enum::complex_f32; };
    template<> struct type_enum_of<cuDoubleComplex>            { static constexpr type_enum value = type_enum::complex_f64; };

    template<typename T>
    inline constexpr type_enum type_to_enum = type_enum_of<T>::value;

    template<type_enum T> struct type_of_enum;
    template<> struct type_of_enum<type_enum::real_f32>    { using type = float; };
    template<> struct type_of_enum<type_enum::real_f64>    { using type = double; };
    template<> struct type_of_enum<type_enum::complex_f32> { using type = cuFloatComplex; };
    template<> struct type_of_enum<type_enum::complex_f64> { using type = cuDoubleComplex; };

    template<type_enum T>
    using type_of_enum_t = typename type_of_enum<T>::type;

} // namespace cusolverdx::detail
#endif // CUSOLVERDX_DETAIL_ENUM_HPP
