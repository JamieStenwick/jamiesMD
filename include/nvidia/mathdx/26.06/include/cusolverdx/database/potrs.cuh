// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_POTRS_CUH
#define CUSOLVERDX_DATABASE_POTRS_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/database/trsm.cuh"
#include "cusolverdx/database/indexing.cuh"

namespace cusolverdx::detail::potrs {
    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, unsigned K, int Arch) {
        return trsm::suggested_batches(T, N, K, side::left, Arch);
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, unsigned K, unsigned BPB, int Arch) {
        return trsm::suggested_block_dim(T, N, K, side::left, BPB, Arch);
    }

    template<class T, unsigned N, unsigned K, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, unsigned NT, unsigned BPB>
    inline __device__ void block_execute(const T* A, const unsigned lda, T* B, const unsigned ldb, const unsigned thread_id) {

        constexpr transpose trans1 = (Fill == fill_mode::lower) ? transpose::non_transposed : transpose::conj_transposed;
        constexpr transpose trans2 = (Fill == fill_mode::lower) ? transpose::conj_transposed : transpose::non_transposed;

        trsm::block_execute<T, N, K, side::left, diag::non_unit, trans1, Fill, ArrangeA, ArrangeB, NT, BPB>(A, lda, B, ldb, thread_id);
        trsm::block_execute<T, N, K, side::left, diag::non_unit, trans2, Fill, ArrangeA, ArrangeB, NT, BPB>(A, lda, B, ldb, thread_id);
    }

    template<class T, unsigned N, unsigned K, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB>
    inline __device__ void thread_execute(const T* A, const unsigned lda, T* B, const unsigned ldb) {

        constexpr transpose trans1 = (Fill == fill_mode::lower) ? transpose::non_transposed : transpose::conj_transposed;
        constexpr transpose trans2 = (Fill == fill_mode::lower) ? transpose::conj_transposed : transpose::non_transposed;

        trsm::thread_execute<T, N, K, side::left, diag::non_unit, trans1, Fill, ArrangeA, ArrangeB>(A, lda, B, ldb);
        trsm::thread_execute<T, N, K, side::left, diag::non_unit, trans2, Fill, ArrangeA, ArrangeB>(A, lda, B, ldb);
    }
} // namespace cusolverdx::detail::potrs

#endif // CUSOLVERDX_DATABASE_POTRS_CUH
