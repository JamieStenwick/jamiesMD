// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GELS_CUH
#define CUSOLVERDX_DATABASE_GELS_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/detail/util.hpp"
#include "cusolverdx/database/geqrf.cuh"
#include "cusolverdx/database/gelqf.cuh"
#include "cusolverdx/database/geqrs.cuh"
#include "cusolverdx/database/util.cuh"

namespace cusolverdx::detail::gels {

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned M, unsigned N, int Arch) {
        // Choice of QR or LQ in GELS is based on which dim is larger
        unsigned eff_m = const_max(M, N);
        unsigned eff_n = const_min(N, M);
        return geqrs::suggested_batches(T, eff_m, eff_n, Arch);
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned M, unsigned N, unsigned BPB, int Arch) {
        // Choice of QR or LQ in GELS is based on which dim is larger
        unsigned eff_m = const_max(M, N);
        unsigned eff_n = const_min(N, M);
        return geqrf::suggested_block_dim(T, eff_m, eff_n, BPB, Arch);
    }

    template<class T, unsigned M, unsigned N, unsigned NRHS, arrangement ArrangeA, arrangement ArrangeB, transpose Trans, unsigned NT, unsigned BPB>
    inline __device__ void block_execute(T* A, const unsigned lda, T* tau, T* B, const unsigned ldb, const unsigned thread_id) {
        // Use QR for tall-skinny and LQ for short-wide
        if constexpr (M >= N) {
            geqrf::block_execute<T, M, N, ArrangeA, NT, BPB>(A, lda, tau, thread_id);
            geqrs::block_execute<T, M, N, NRHS, ArrangeA, ArrangeB, Trans, NT, BPB>(A, lda, tau, B, ldb, thread_id);
        } else {
            gelqf::block_execute<T, M, N, ArrangeA, NT, BPB>(A, lda, tau, thread_id);
            gelqs::block_execute<T, M, N, NRHS, ArrangeA, ArrangeB, Trans, NT, BPB>(A, lda, tau, B, ldb, thread_id);
        }
    }

    template<class T, unsigned M, unsigned N, unsigned NRHS, arrangement ArrangeA, arrangement ArrangeB, transpose Trans>
    inline __device__ void thread_execute(T* A, const unsigned lda, T* tau, T* B, const unsigned ldb) {
        // Use QR for tall-skinny and LQ for short-wide
        if constexpr (M >= N) {
            geqrf::thread_execute<T, M, N, ArrangeA>(A, lda, tau);
            geqrs::thread_execute<T, M, N, NRHS, ArrangeA, ArrangeB, Trans>(A, lda, tau, B, ldb);
        } else {
            gelqf::thread_execute<T, M, N, ArrangeA>(A, lda, tau);
            gelqs::thread_execute<T, M, N, NRHS, ArrangeA, ArrangeB, Trans>(A, lda, tau, B, ldb);
        }
    }

} // namespace cusolverdx::detail::gels

#endif // CUSOLVERDX_DATABASE_GELS_CUH
