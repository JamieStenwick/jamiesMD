#ifndef CURANDDX_DETAIL_COMMON_HPP
#define CURANDDX_DETAIL_COMMON_HPP

#if !defined(CURANDDX_QUALIFIERS)
#    define CURANDDX_QUALIFIERS __forceinline__ __device__
#endif

#include "commondx/detail/stl/type_traits.hpp"

namespace curanddx {
    namespace detail {
        template<typename T = float>
        inline constexpr T rand_2pow32_inv = T(2.3283064365386963e-10);

        inline constexpr double rand_2pow53_inv_double = 1.1102230246251565e-16;

        template<typename T, unsigned int N>
        struct make_vector {
            using type = void;
        };
        template<typename T>
        struct make_vector<T, 1> {
            using type = T;
        };
        template<>
        struct make_vector<unsigned int, 2> {
            using type = uint2;
        };
        template<>
        struct make_vector<unsigned int, 4> {
            using type = uint4;
        };
        template<>
        struct make_vector<unsigned long long, 2> {
            using type = ulonglong2;
        };
        template<>
        struct make_vector<unsigned long long, 4> {
            using type = ulonglong4;
        };
        template<>
        struct make_vector<float, 2> {
            using type = float2;
        };
        template<>
        struct make_vector<float, 4> {
            using type = float4;
        };
        template<>
        struct make_vector<double, 2> {
            using type = double2;
        };
        template<>
        struct make_vector<double, 4> {
            using type = double4;
        };

    } // namespace detail
} // namespace curanddx


#endif
