// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_TRSM_HPP
#define CUBLASDX_DATABASE_TRSM_HPP

#include "cublasdx/types.hpp"
#include "cublasdx/database/trsm/trsm_db.hpp"

namespace cublasdx::detail::trsm {

    // Scalar one: T(1) for real, T(1,0) for complex.
    template<class T> CUBLASDX_HOST_DEVICE T trsm_one() { return T(1); }
    template<> CUBLASDX_HOST_DEVICE cublasdx::complex<float>  trsm_one<cublasdx::complex<float>>()  { return {1.f, 0.f}; }
    template<> CUBLASDX_HOST_DEVICE cublasdx::complex<double> trsm_one<cublasdx::complex<double>>() { return {1.0, 0.0}; }

    // Scalar zero: T(0) for real, T(0,0) for complex.
    template<class T> CUBLASDX_HOST_DEVICE T trsm_zero() { return T(0); }
    template<> CUBLASDX_HOST_DEVICE cublasdx::complex<float>  trsm_zero<cublasdx::complex<float>>()  { return {0.f, 0.f}; }
    template<> CUBLASDX_HOST_DEVICE cublasdx::complex<double> trsm_zero<cublasdx::complex<double>>() { return {0.0, 0.0}; }

    template<class T>
    using tensor_readonly_accessor = T (*const)(const void*, int, int);

    template<class T>
    using tensor_readwrite_accessor = T&(*const)(void*, int, int);

    // Accessor-based compiled fatbin functions.
    // All layout/fill/conjugation logic is baked into the accessor lambdas.
    template<class T>
    __device__ void thread_impl(void const* const tensor_a_ptr, void* const tensor_b_ptr,
                                T* B_local, unsigned M, unsigned N,
                                bool unit_diag, bool forward,
                                const tensor_readonly_accessor<T> tensor_accessor_a, 
                                const tensor_readwrite_accessor<T> tensor_accessor_b);
    // Designed for small N
    template<class T>
    __device__ void warp_impl(void const* const tensor_a_ptr, void* const tensor_b_ptr,
                              unsigned thread_id, T* B_local, T* B_i, T* A_diag,
                              unsigned NT, unsigned M, unsigned N,
                              bool unit_diag, bool forward,
                              const tensor_readonly_accessor<T> tensor_accessor_a, 
                              const tensor_readwrite_accessor<T> tensor_accessor_b);
    // Designed for small N
    template<class T>
    __device__ void cta_impl(void const* const tensor_a_ptr, void* const tensor_b_ptr,
                             unsigned thread_id, T* B_local, T* B_i,
                             unsigned M, unsigned N, unsigned NT,
                             bool unit_diag, bool forward,
                             const tensor_readonly_accessor<T> tensor_accessor_a, 
                             const tensor_readwrite_accessor<T> tensor_accessor_b);

    // Returns the largest power-of-2 that divides n (lowest set bit).
    constexpr __device__ __forceinline__ unsigned pow2_divisor(unsigned n) { return n & (~n + 1u); }

    // ------------------------------------------------------------------
    // Accessor factory functions.
    //
    // Right-side TRSM solves with A^T and B^T, so both A and B need their
    // indices swapped (i,j)↔(j,i) compared to left-side.  The fatbin's
    // `fwd` flag already encodes which triangle to iterate; no Fill-based
    // index swap is needed in the accessor.
    // ------------------------------------------------------------------

    template<class Tensor, class T = typename Tensor::value_type>
    __device__ __forceinline__ T& trsm_mutable_tensor_accessor(void* t_ptr, int i, int j) {
        const auto& t = *reinterpret_cast<const Tensor*>(t_ptr);
        return t(i, j);
    }

    template<class Tensor, class T = typename Tensor::value_type>
    __device__ __forceinline__ T& trsm_mutable_vector_accessor(void* t_ptr, int i, int) {
        const auto& t = *reinterpret_cast<const Tensor*>(t_ptr);
        return t(i);
    }

    template<class Tensor, class T = typename Tensor::value_type>
    __device__ __forceinline__ T trsm_immutable_tensor_accessor(const void* t_ptr, int i, int j) {
        const auto& t = *reinterpret_cast<const Tensor*>(t_ptr);
        return t(i, j);
    }

    // block_execute and thread_execute are defined in trsm_execute.hpp,
    // which also includes cublasdx/detail/tensor.hpp (needed for cublasdx::tensor<>).
    // They are kept out of this file so that this header remains includable
    // from the fatbin compilation path (libcublasdx.cu → trsm_dispatch.hpp).

} // namespace cublasdx::detail::trsm

#endif // CUBLASDX_DATABASE_TRSM_HPP
