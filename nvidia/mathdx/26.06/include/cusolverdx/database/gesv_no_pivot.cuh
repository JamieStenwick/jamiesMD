// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GESV_NO_PIVOT_CUH
#define CUSOLVERDX_DATABASE_GESV_NO_PIVOT_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/getrf_no_pivot.cuh"
#include "cusolverdx/database/getrs_no_pivot.cuh"
#include "cusolverdx/detail/block_execution_bugs.hpp"

namespace cusolverdx::detail::gesv_no_pivot {

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, int Arch) {
        return getrf_no_pivot::suggested_batches(T, N, N, Arch);
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, unsigned BPB, int Arch) {
        return getrf_no_pivot::suggested_block_dim(T, N, N, BPB, Arch);
    }

    template<class T, unsigned N, unsigned NRHS, arrangement ArrangeA, arrangement ArrangeB, transpose Trans, unsigned NT, unsigned BPB, int Arch>
    inline __device__ void block_execute(T* A, const unsigned lda, T* B, const unsigned ldb, int* info, const unsigned thread_id) {
        CUSOLVERDX_NVBUG_5288270_GESV_NO_PIVOT_HANDLER(Arch, is_real_v<T>);

        getrf_no_pivot::block_execute<T, N, N, ArrangeA, NT, BPB, Arch>(A, lda, info, thread_id);
        getrs_no_pivot::block_execute<T, N, NRHS, ArrangeA, ArrangeB, Trans, NT, BPB>(A, lda, B, ldb, thread_id);
    }

    template<class T, unsigned N, unsigned NRHS, arrangement ArrangeA, arrangement ArrangeB, transpose Trans>
    inline __device__ void thread_execute(T* A, const unsigned lda, T* B, const unsigned ldb, int* info) {
        getrf_no_pivot::thread_execute<T, N, N, ArrangeA>(A, lda, info);
        getrs_no_pivot::thread_execute<T, N, NRHS, ArrangeA, ArrangeB, Trans>(A, lda, B, ldb);
    }

} // namespace cusolverdx::detail::gesv_no_pivot

#endif // CUSOLVERDX_DATABASE_GESV_NO_PIVOT_CUH
