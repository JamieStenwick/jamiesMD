// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HEGST_CUH
#define CUSOLVERDX_DATABASE_HEGST_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/detail/type_enum.hpp"

#include "cusolverdx/database/hegst_db.cuh"
#include "cusolverdx/database/grid_utils.cuh"

namespace cusolverdx::detail::hegst {

    // hegst has 2 modes:
    // * t1 (itype == 1)    which computes L^-1 A L^-H or U^-H A U^-1
    // * t2 (itype == 2, 3) which computes L^H A L or U A U^H

    template<class T>
    __device__ void thread_driver(T* A, int lda, const T* B, int ldb, unsigned tid, unsigned N, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool is_t1, unsigned NT, unsigned BPB, T* rwork);
    template<class T>
    __device__ void warp_driver(T* A, int lda, const T* B, int ldb, unsigned tid, unsigned N, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool is_t1, unsigned NT, unsigned BPB, int p, int q, T* rwork);

    template<class T>
    __device__ void cta_driver(T* A, int lda, const T* B, int ldb, unsigned tid, unsigned N, const fill_mode Fill, const arrangement ArrangeA, const arrangement ArrangeB, const bool is_t1, unsigned NT, unsigned BPB, int p, int q, T* rwork, T* comm_work);

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, [[maybe_unused]] int itype, int Arch) {
        auto thresholds = suggested_bpb_size_thresholds(T, Arch);
        if (N <= thresholds[0]) {
            return 32;
        } else if (N <= thresholds[1]) {
            return 16;
        } else if (N <= thresholds[2]) {
            return 8;
        } else if (N <= thresholds[3]) {
            return 4;
        } else if (N <= thresholds[4]) {
            return 2;
        } else {
            return 1;
        }
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, [[maybe_unused]] int itype, unsigned BPB, int Arch) {
        if (BPB > 1u) {
            // TODO review this choice
            return ((BPB+32-1)/32)*32;
        } else {
            auto thresholds = suggested_block_dim_size_thresholds(T, Arch);
            if (N <= thresholds[0]) {
                return 32;
            } else if (N <= thresholds[1]) {
                return 64;
            } else if (N <= thresholds[2]) {
                return 128;
            } else if (N <= thresholds[3]) {
                return 256;
            } else if (N <= thresholds[4]) {
                return 512;
            } else {
                return 1024;
            }
        }
    }

    // TODO tune this dispatch
    constexpr inline __device__ __host__ bool use_thread_per_batch(type_enum T, unsigned N, [[maybe_unused]] int itype, unsigned NT, unsigned BPB, int Arch) {
        return N <= tiny_threshold(T, Arch) || NT % 32 != 0 || BPB > NT / 2;
    }

    constexpr inline __device__ __host__ bool use_warp_per_batch([[maybe_unused]] type_enum T, [[maybe_unused]] unsigned N, [[maybe_unused]] int itype, unsigned NT, unsigned BPB, [[maybe_unused]] int Arch) {
        return BPB >= (NT / 32) && NT % 32 == 0;
    }

    // Number of words of type T needed as shared-memory workspace
    // Only needed for the cta-driver, so thread and warp drivers request 0
    constexpr inline __device__ __host__ int workspace_size(type_enum T, unsigned N, int itype, unsigned NT, unsigned BPB, int Arch) {
        if (use_thread_per_batch(T, N, itype, NT, BPB, Arch)
                      || use_warp_per_batch(T, N, itype, NT, BPB, Arch)) {
            return 0;
        } else {
            // p*q == NT; row reductions and broadcasts use up to NT words of T
            return NT;
        }
    }
    constexpr inline __device__ __host__ int workspace_size_thread([[maybe_unused]] type_enum T, [[maybe_unused]] unsigned N, [[maybe_unused]] int itype) {
        return 0;
    }


    // Fill is currently assumed to be the same for A and B
    // comm_work: shared-memory buffer of at least workspace_size(T,N,itype,NT,BPB,Arch) elements; required for the CTA path, ignored otherwise.
    template<class T, unsigned N, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, int itype, unsigned NT, unsigned BPB, int Arch>
    __device__ void block_execute(T* A, int lda, const T* B, int ldb, [[maybe_unused]] T* comm_work, unsigned tid) {
        static_assert(itype == 1 || itype == 2 || itype == 3, "itype for generalized eigensolvers are only 1, 2, or 3");

        constexpr bool is_t1 = itype == 1;

        if constexpr (use_thread_per_batch(type_to_enum<T>, N, itype, NT, BPB, Arch)) {
            T rwork[2*N*N];
            thread_driver<T>(A, lda, B, ldb, tid, N, Fill, ArrangeA, ArrangeB, is_t1, NT, BPB, rwork);

        } else if constexpr (use_warp_per_batch(type_to_enum<T>, N, itype, NT, BPB, Arch)) {

            constexpr unsigned subwarp_size = compute_subwarp_size<BPB, NT>();
            constexpr auto pq = pq_selector(N, N, subwarp_size);
            constexpr auto p = pq.p;
            constexpr auto q = pq.q;
            constexpr int nrows = (N - 1) / p + 1;
            constexpr int ncols = (N - 1) / q + 1;

            T rwork[2 * nrows * ncols + 2 * nrows + 2 * ncols];
            warp_driver<T>(A, lda, B, ldb, tid, N, Fill, ArrangeA, ArrangeB, is_t1, NT, BPB, p, q, rwork);

        } else {
            // 2D cyclic grid: p*q == NT (match warp path factorization when NT is a multiple of 8)
            constexpr auto pq = pq_selector(N, N, NT);
            constexpr auto p = pq.p;
            constexpr auto q = pq.q;
            constexpr int nrows = (N - 1) / p + 1;
            constexpr int ncols = (N - 1) / q + 1;
            T rwork[2 * nrows * ncols + 2 * nrows + 2 * ncols];
            cta_driver<T>(A, lda, B, ldb, tid, N, Fill, ArrangeA, ArrangeB, is_t1, NT, BPB, p, q, rwork, comm_work);
        }
    }

    template<class T, unsigned N, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, int itype>
    __device__ void thread_execute(T* A, int lda, const T* B, int ldb, [[maybe_unused]] T* comm_work) {
        static_assert(itype == 1 || itype == 2 || itype == 3, "itype for generalized eigensolvers are only 1, 2, or 3");

        bool is_t1 = itype == 1;

        T rwork[2*N*N];
        thread_driver<T>(A, lda, B, ldb, 0, N, Fill, ArrangeA, ArrangeB, is_t1, 1, 1, rwork);
    }

} // namespace cusolverdx::detail::hegst

#endif // CUSOLVERDX_DATABASE_HEGST_CUH
