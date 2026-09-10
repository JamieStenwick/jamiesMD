// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_OPERATORS_GENERATOR_HPP
#define CURANDDX_OPERATORS_GENERATOR_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace curanddx {

    enum class generator
    {
        xorwow,
        mrg32k3a,
        philox4_32,
        mtgp32,
        pcg,
        sobol32,
        scrambled_sobol32,
        sobol64,
        scrambled_sobol64
    };
    inline constexpr auto xorwow            = generator::xorwow;
    inline constexpr auto mrg32k3a          = generator::mrg32k3a;
    inline constexpr auto philox4_32        = generator::philox4_32;
    inline constexpr auto mtgp32            = generator::mtgp32;
    inline constexpr auto pcg               = generator::pcg;
    inline constexpr auto sobol32           = generator::sobol32;
    inline constexpr auto scrambled_sobol32 = generator::scrambled_sobol32;
    inline constexpr auto sobol64           = generator::sobol64;
    inline constexpr auto scrambled_sobol64 = generator::scrambled_sobol64;

    template<generator Value>
    struct Generator: public commondx::detail::constant_operator_expression<generator, Value> {};

    namespace detail {
        using default_curanddx_generator_operator = Generator<philox4_32>;

        template<curanddx::generator Value>
        struct is_philox: COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value == philox4_32)> {};

        template<curanddx::generator Value>
        struct is_pseudo_random:
            COMMONDX_STL_NAMESPACE::integral_constant<
                bool,
                (Value == xorwow || Value == mrg32k3a || Value == mtgp32 || Value == pcg) || is_philox<Value>::value> {
        };

        template<curanddx::generator Value>
        struct is_sobol: COMMONDX_STL_NAMESPACE::integral_constant<bool, !is_pseudo_random<Value>::value> {};

        // Consider PCG 32-bit generator as well
        template<curanddx::generator Value>
        struct is_32bits:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, (Value != sobol64 && Value != scrambled_sobol64)> {};

        template<curanddx::generator Value>
        struct is_64bits:
            COMMONDX_STL_NAMESPACE::
                integral_constant<bool, (Value == sobol64 || Value == scrambled_sobol64)> {};
    } // namespace detail

} // namespace curanddx

namespace commondx::detail {
    template<curanddx::generator Value>
    struct is_operator<curanddx::operator_type, curanddx::operator_type::generator, curanddx::Generator<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {};

    template<curanddx::generator Value>
    struct get_operator_type<curanddx::operator_type, curanddx::Generator<Value>> {
        static constexpr curanddx::operator_type value = curanddx::operator_type::generator;
    };
} // namespace commondx::detail

#endif // CURANDDX_OPERATORS_GENERATOR_HPP
