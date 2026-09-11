// Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_TYPES_HPP
#define CUBLASDX_TYPES_HPP

#include "commondx/complex_types_cutlass.hpp"
#include "commondx/types.hpp"

#ifndef CUBLASDX_HOST_DEVICE
#    define CUBLASDX_HOST_DEVICE __host__ __device__ __forceinline__
#endif

#ifndef CUBLASDX_HOST
#    define CUBLASDX_HOST __host__
#endif

#ifndef CUBLASDX_DEVICE
#    define CUBLASDX_DEVICE __device__ __forceinline__
#endif

namespace cublasdx {
    namespace detail {
        template<class T, typename = void>
        struct has_complex_interface: COMMONDX_STL_NAMESPACE::false_type {
        };

        template<class T>
        struct has_complex_interface<
            T,
            COMMONDX_STL_NAMESPACE::void_t<decltype(COMMONDX_STL_NAMESPACE::declval<T const&>().real()),
                                           decltype(COMMONDX_STL_NAMESPACE::declval<T const&>().imag()),
                                           typename T::value_type>>: COMMONDX_STL_NAMESPACE::true_type {
        };

        template<class T>
        inline constexpr bool has_complex_interface_v = has_complex_interface<T>::value;

        template<typename T, typename = void>
        struct convert_to_cutlass_type {
            using type = T;
        };

        template<>
        struct convert_to_cutlass_type<__half> {
            using type = cutlass::half_t;
            static_assert((alignof(__half) >= alignof(type)), "cutlass type has stricter alignment requirement.");
        };

        template<>
        struct convert_to_cutlass_type<__nv_bfloat16> {
            using type = cutlass::bfloat16_t;
            static_assert((alignof(__nv_bfloat16) >= alignof(type)),
                          "cutlass type has stricter alignment requirement.");
        };

#ifdef COMMONDX_DETAIL_CUDA_FP8_ENABLED
        template<>
        struct convert_to_cutlass_type<__nv_fp8_e5m2> {
            using type = cutlass::float_e5m2_t;
            static_assert((alignof(__nv_fp8_e5m2) >= alignof(type)),
                          "cutlass type has stricter alignment requirement.");
        };

        template<>
        struct convert_to_cutlass_type<__nv_fp8_e4m3> {
            using type = cutlass::float_e4m3_t;
            static_assert((alignof(__nv_fp8_e4m3) >= alignof(type)),
                          "cutlass type has stricter alignment requirement.");
        };
#endif

        template<typename T>
        struct convert_to_cutlass_type<T, COMMONDX_STL_NAMESPACE::enable_if_t<has_complex_interface_v<T>>> {
            using internal_value_type = typename T::value_type;
            using value_type          = typename convert_to_cutlass_type<internal_value_type>::type;
            using type                = cutlass::complex<value_type>;
            static_assert((alignof(T) >= alignof(type)), "cutlass type has stricter alignment requirement.");
        };

        template<class T>
        using convert_to_cutlass_type_t = typename convert_to_cutlass_type<T>::type;

        template<class CT, class T>
        __forceinline__ __device__ __host__ COMMONDX_STL_NAMESPACE::enable_if_t<cutlass::is_complex<CT>::value, CT>
                                            cast_to_cutlass_type(const T a) {
            using value_type = typename CT::value_type;
            return CT(static_cast<value_type>(a.real()), static_cast<value_type>(a.imag()));
        }

        template<class CT, class T>
        __forceinline__ __device__ __host__ COMMONDX_STL_NAMESPACE::enable_if_t<!cutlass::is_complex<CT>::value, CT>
                                            cast_to_cutlass_type(const T a) {
            return static_cast<CT>(a);
        }

        __host__ __device__ __forceinline__ __nv_bfloat16 ushort_as_bfloat(unsigned short value) {
#if defined(__CUDA_ARCH__)
            return reinterpret_cast<__nv_bfloat16&>(value);
#else
            __nv_bfloat16 raw;
            COMMONDX_STL_NAMESPACE::memcpy(
                reinterpret_cast<commondx::byte*>(&raw), reinterpret_cast<commondx::byte*>(&value), sizeof(value));
            return raw;
#endif
        }

        template<class T, class CT = convert_to_cutlass_type_t<T>>
        __forceinline__ __device__ __host__ COMMONDX_STL_NAMESPACE::enable_if_t<cutlass::is_complex<CT>::value, T>
                                            cast_from_cutlass_type(const CT a) {
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, typename T::value_type>) {
                return T(a.real(), a.imag());
            } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, cutlass::half_t>) {
                return T(a.real().to_half(), a.imag().to_half());
            } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, cutlass::bfloat16_t>) {
                return T(ushort_as_bfloat(a.real().raw()), ushort_as_bfloat(a.imag().raw()));
            }
#if COMMONDX_DETAIL_CUDA_FP8_ENABLED
            else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, cutlass::float_e5m2_t> ||
                               COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, cutlass::float_e4m3_t>) {
                using value_type = typename T::value_type;
                return T(static_cast<value_type>(float(a.real())), static_cast<value_type>(float(a.imag())));
            }
#endif
            else {
                using value_type = typename T::value_type;
                return T(static_cast<value_type>(a.real()), static_cast<value_type>(a.imag()));
            }

            CUTE_GCC_UNREACHABLE;
        }

        template<class T, class CT = convert_to_cutlass_type_t<T>>
        __forceinline__ __device__ __host__ COMMONDX_STL_NAMESPACE::enable_if_t<!cutlass::is_complex<CT>::value, T>
                                            cast_from_cutlass_type(CT a) {
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CT, T>) {
                return a;
            } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CT, cutlass::half_t>) {
                return a.to_half();
            } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CT, cutlass::bfloat16_t>) {
                return ushort_as_bfloat(a.raw());
            }
#if COMMONDX_DETAIL_CUDA_FP8_ENABLED
            else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CT, cutlass::float_e5m2_t> ||
                               COMMONDX_STL_NAMESPACE::is_same_v<CT, cutlass::float_e4m3_t>) {
                return static_cast<T>(float(a));
            }
#endif
            else {
                return static_cast<T>(a);
            }

            CUTE_GCC_UNREACHABLE;
        }
    } // namespace detail

    using commondx::bfloat16_t;
    using commondx::complex;
    using commondx::float_e4m3_t;
    using commondx::float_e5m2_t;
    using commondx::half_t;
    using commondx::tfloat32_t;
    using commondx::byte;

} // namespace cublasdx

#endif // CUBLASDX_TYPES_HPP
