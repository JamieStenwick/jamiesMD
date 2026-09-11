// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_UTIL_CUH
#define CUSOLVERDX_DATABASE_UTIL_CUH

#include "cusolverdx/operators/enums.hpp"
#include "cuComplex.h"
#ifndef CUSOLVERDX_NO_COMMONDX_COMPLEX
    #include "cusolverdx/types.hpp"
#endif // CUSOLVERDX_NO_COMMONDX_COMPLEX

namespace cusolverdx {
    namespace detail {

        // Implementation of is_same to avoid needing type_traits
        template<class T, class U>
        struct same_type {
            static constexpr bool value = false;
        };
        template<class T>
        struct same_type<T, T> {
            static constexpr bool value = true;
        };
        template<class T, class U>
        constexpr bool same_type_v = same_type<T, U>::value;


        template<class T>
        struct is_real {
            static constexpr bool value = true;
        };
        template<>
        struct is_real<cuComplex> {
            static constexpr bool value = false;
        };
        template<>
        struct is_real<cuDoubleComplex> {
            static constexpr bool value = false;
        };
        #ifndef CUSOLVERDX_NO_COMMONDX_COMPLEX
            template<>
            struct is_real<commondx::complex<float>> {
                static constexpr bool value = false;
            };
            template<>
            struct is_real<commondx::complex<double>> {
                static constexpr bool value = false;
            };
        #endif // CUSOLVERDX_NO_COMMONDX_COMPLEX

        template<class T>
        static constexpr bool is_real_v = is_real<T>::value;

        template<class T>
        struct real_type {
            using value_type = T;
        };
        template<>
        struct real_type<cuComplex> {
            using value_type = float;
        };
        template<>
        struct real_type<cuDoubleComplex> {
            using value_type = double;
        };
        #ifndef CUSOLVERDX_NO_COMMONDX_COMPLEX
            template<>
            struct real_type<commondx::complex<float>> {
                using value_type = float;
            };
            template<>
            struct real_type<commondx::complex<double>> {
                using value_type = double;
            };
        #endif // CUSOLVERDX_NO_COMMONDX_COMPLEX
        template<class T>
        using real_type_t = typename real_type<T>::value_type;

        template<class T>
        inline __device__ T scalar(real_type_t<T> value) {
            if constexpr (is_real_v<T>) {
                return value;
            } else {
                return T {value, real_type_t<T>(0)};
            }
        }

        template<class T>
        constexpr __device__ __host__ T const_max(T a, T b) {
            return a >= b ? a : b;
        }

        // Computes the largest power of 2 that divides x
        constexpr __device__ __host__ int pow2_divisor(int x) {
            return x & (~(x - 1));
        }

        // Compute the smallest power of 2 greater than i
        // Assumes 1 <= i <= 0x7FFFFFFF
        constexpr __device__ __host__ unsigned smallest_pow2_greater(unsigned i) {
            static_assert(sizeof(i) == 4);

            i -= 1;
            i |= i >> 1;
            i |= i >> 2;
            i |= i >> 3;
            i |= i >> 4;
            i |= i >> 8;
            i |= i >> 16;
            return i + 1;
        }

        inline __device__ bool is_stored_triangle(const unsigned row, const unsigned col, const fill_mode Fill) {
            return Fill == fill_mode::lower ? row >= col : row <= col;
        }

        template<int N>
        struct threshold_array {
            unsigned vals[N];

            constexpr __device__ __host__ unsigned operator[](size_t i) {
                return vals[i];
            }
        };

    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_UTIL_CUH
