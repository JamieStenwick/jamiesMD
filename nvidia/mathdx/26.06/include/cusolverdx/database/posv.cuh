// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_POSV_CUH
#define CUSOLVERDX_DATABASE_POSV_CUH

#include "cusolverdx/database/potrf.cuh"
#include "cusolverdx/database/potrs.cuh"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx::detail::posv {

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, int Arch) {
        return potrf::suggested_batches(T, N, Arch);
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, unsigned BPB, int Arch) {
        return potrf::suggested_block_dim(T, N, BPB, Arch);
    }

    template<class T, unsigned N, unsigned NRHS, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned BPB, int Arch>
    inline __device__ void block_execute(T* A, const unsigned lda, T* B, const unsigned ldb, int* info, const unsigned thread_id) {
        potrf::block_execute<T, N, Fill, ArrangeA, NT, BPB, Arch>(A, lda, info, thread_id);
        potrs::block_execute<T, N, NRHS, Fill, ArrangeA, ArrangeB, NT, BPB>(A, lda, B, ldb, thread_id);
    }

    template<class T, unsigned N, unsigned NRHS, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB>
    inline __device__ void thread_execute(T* A, const unsigned lda, T* B, const unsigned ldb, int* info) {
        potrf::thread_execute<T, N, Fill, ArrangeA>(A, lda, info);
        potrs::thread_execute<T, N, NRHS, Fill, ArrangeA, ArrangeB>(A, lda, B, ldb);
    }

} // namespace cusolverdx::detail::posv

#endif // CUSOLVERDX_DATABASE_POSV_CUH
