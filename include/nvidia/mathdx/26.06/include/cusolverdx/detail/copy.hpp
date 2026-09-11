// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_COPY_HPP
#define CUSOLVERDX_DETAIL_COPY_HPP

#include <cute/tensor.hpp>
#include "cusolverdx/traits/solver_traits.hpp"

namespace cusolverdx {
    namespace detail {
        template<class T>
        __device__ __forceinline__ auto make_cute_compatible_ptr(T* ptr) {
            // This is necessary because cute::cooperative_copy accepts either smem
            // or gmem pointers only. The GMEM tag can be used for either and won't 
            // cause a dispatch to any memory space dedicated instructions.
            return cute::make_gmem_ptr(ptr);
        }

        template<unsigned int M, unsigned int N, unsigned int Batches, cusolverdx::arrangement Arrange>
        static inline __device__ auto get_layout(unsigned int ld) {
            const auto shape = cute::make_shape(cute::Int<M>{}, cute::Int<N>{}, cute::Int<Batches>{});
            constexpr auto slow_dim = cute::conditional_return<Arrange == cusolverdx::col_major>(cute::Int<N>{}, cute::Int<M>{});
            const auto stride = cute::conditional_return<Arrange == cusolverdx::col_major>(cute::make_stride(cute::_1{}, ld, slow_dim * ld), 
                                                                                            cute::make_stride(ld, cute::_1{}, slow_dim * ld));
            return cute::make_layout(shape, stride);
        }

        template<unsigned int N, unsigned int Batches>
        static inline __device__ auto get_layout() {
            const auto shape = cute::make_shape(cute::Int<N>{}, cute::Int<Batches>{});
            const auto stride = cute::make_stride(cute::_1{}, cute::Int<N>{});
            return cute::make_layout(shape, stride);
        }
    } // namespace detail

    template<int Threads, unsigned int M, unsigned int N, cusolverdx::arrangement Arrange, unsigned int Batches = 1, class DataType>
    static inline __device__ void copy_2d(const DataType* src, const int ld_src, DataType* dst, const int ld_dst) {
        const auto tid     = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
        const auto src_tensor = cute::make_tensor(detail::make_cute_compatible_ptr(src), detail::get_layout<M, N, Batches, Arrange>(ld_src));
        auto dst_tensor = cute::make_tensor(detail::make_cute_compatible_ptr(dst), detail::get_layout<M, N, Batches, Arrange>(ld_dst));
        cute::cooperative_copy<Threads, 8 * alignof(DataType)>(tid, src_tensor, dst_tensor);
    }

    template<class Operation, unsigned int M = Operation::m_size, unsigned int N = Operation::n_size, cusolverdx::arrangement Arrange = cusolverdx::arrangement_of_v_a<Operation>, unsigned int Batches = 1, class DataType = typename Operation::a_data_type>
    static inline __device__ void copy_2d(const DataType* src, const int ld_src, DataType* dst, const int ld_dst) {
        copy_2d<Operation::max_threads_per_block, M, N, Arrange, Batches, DataType>(src, ld_src, dst, ld_dst);
    }

} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_COPY_HPP
