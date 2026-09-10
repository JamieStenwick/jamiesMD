// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_TYPES_HPP
#define CUSOLVERDX_TYPES_HPP

#include <cuComplex.h>
#include "commondx/types.hpp"

#ifdef CUSOLVERDX_NO_COMMONDX_COMPLEX
    #error "Invalid combination of headers"
#endif
#include "commondx/complex_types.hpp"

namespace cusolverdx {
    // imported types, aliases and conversion utilities
    template<typename T>
    struct convert_to_cuda_type {
        using type = T;
    };

    template<>
    struct convert_to_cuda_type<commondx::complex<float>> {
        using type = cuFloatComplex;
        static_assert((alignof(commondx::complex<float>) >= alignof(type)),
                      "commondx type has stricter alignment requirement.");
    };

    template<>
    struct convert_to_cuda_type<commondx::complex<double>> {
        using type = cuDoubleComplex;
        static_assert((alignof(commondx::complex<double>) >= alignof(type)),
                      "commondx type has stricter alignment requirement.");
    };

    // cublasdx uses commondx::complex<> and cusolverdx uses cuFloatComplex/cuDoubleComplex
    template<typename T>
    struct convert_to_commondx_type {
        using type = T;
    };

    template<>
    struct convert_to_commondx_type<cuFloatComplex> {
        using type = commondx::complex<float>;
        static_assert(sizeof(type) == sizeof(cuFloatComplex), "commondx complex type must preserve CUDA complex storage size.");
        static_assert(alignof(type) == alignof(cuFloatComplex), "commondx complex type must preserve CUDA complex alignment.");
    };

    template<>
    struct convert_to_commondx_type<cuDoubleComplex> {
        using type = commondx::complex<double>;
        static_assert(sizeof(type) == sizeof(cuDoubleComplex), "commondx complex type must preserve CUDA complex storage size.");
        static_assert(alignof(type) == alignof(cuDoubleComplex), "commondx complex type must preserve CUDA complex alignment.");
    };


    template<class T>
    using complex = commondx::complex<T>;
    using byte    = commondx::byte;
} // namespace cusolverdx

#endif // CUSOLVERDX_TYPES_HPP
