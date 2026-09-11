// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GEBRD_CUH
#define CUSOLVERDX_DATABASE_GEBRD_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/detail/type_enum.hpp"

#include "cusolverdx/database/grid_utils.cuh"
#include "cusolverdx/database/util.cuh"

namespace cusolverdx::detail::gebrd {

    template<class T>
    __device__ void thread_driver(T* A, const unsigned lda, T* tau_left, T* tau_right, const unsigned thread_id, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned NT, const unsigned Batches, T* rwork);
    template<class T>
    __device__ void warp_driver(T* A, const unsigned lda, T* tau_left, T* tau_right, const unsigned lane_id, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned p, const unsigned q, const unsigned num_warps, const unsigned warp_id, const unsigned Batches, T* rwork);
    template<class T, bool big_size>
    __device__ void cta_driver(T* A, const unsigned lda, T* tau_left, T* tau_right, const unsigned thread_id, const unsigned M, const unsigned N, const arrangement Arrange, const unsigned p, const unsigned q, const unsigned Batches, T* rwork, T* swork);

    // TODO need to tune the dispatch
    constexpr inline __device__ __host__ bool use_thread_per_batch([[maybe_unused]] type_enum T, unsigned M, unsigned N, unsigned NT, [[maybe_unused]] unsigned BPB, [[maybe_unused]] int Arch) {
        return M*N <= 64u || NT < 32u;
    }

    constexpr inline __device__ __host__ bool use_warp_per_batch([[maybe_unused]] type_enum T, unsigned M, unsigned N, unsigned NT, [[maybe_unused]] unsigned BPB, [[maybe_unused]] int Arch) {
        return M*N <= 32u*32u     // Moderate problem sizes
                || NT == 32u      // The warp impl is always better than the CTA impl when there is exactly 1 warp
                || NT % 32u != 0; // The CTA impl doesn't work with irregular NT
    }

    constexpr inline __device__ __host__ bool use_cta_big_per_batch(type_enum T, unsigned M, unsigned N, [[maybe_unused]] unsigned NT, [[maybe_unused]] unsigned BPB, [[maybe_unused]] int Arch) {
        // CTA impl has some optimizations for big problems.  Flag such problems
        if (T == type_enum::real_f32) {
            return M*N >= 76u*76u;
        } else if (T == type_enum::real_f64) {
            return M*N >= 56u*56u;
        } else if (T == type_enum::complex_f32) { //complex<float>
            return M*N >= 48u*48u;
        } else { // complex<double>
            return M*N >= 44u*44u;
        }
    }


    // Number of words of type T needed as workspace
    constexpr inline __device__ __host__ int workspace_size(type_enum T, unsigned M, unsigned N, unsigned NT, unsigned BPB, int Arch) {
        if (use_thread_per_batch(T, M, N, NT, BPB, Arch) || use_warp_per_batch(T, M, N, NT, BPB, Arch)) {
            return 0;
        } else {
            return NT;
        }
    }

    // Batches of A have the normal stride for an MxN matrix of Arrange with lda leading dimension
    // Batches of tau have a stride of min(M, N)
    template<class T, unsigned M, unsigned N, arrangement Arrange, unsigned NT, unsigned BPB, unsigned Arch>
    __device__ void block_execute(T* A, int lda, T* tau_left, T* tau_right, unsigned tid, T* swork) {
        static_assert(M >= N, "gebrd doesn't support M < N");

        if constexpr (use_thread_per_batch(type_to_enum<T>, M, N, NT, BPB, Arch)) {
            T rwork[M*N];
            thread_driver<T>(A, lda, tau_left, tau_right, tid, M, N, Arrange, NT, BPB, rwork);
        } else if constexpr (use_warp_per_batch(type_to_enum<T>, M, N, NT, BPB, Arch)) {
            constexpr int num_warps = NT / 32;
            int warp_id = tid / 32;
            int lane_id = tid % 32;

            constexpr auto raw_pq = pq_selector(M, N, 32);
            constexpr int p = raw_pq.p;
            constexpr int q = raw_pq.q;

            T rwork[M*N + 2*M];

            // TODO tighten up workspace size
            if (NT % 32 == 0 || warp_id < num_warps) {
                warp_driver<T>(A, lda, tau_left, tau_right, lane_id, M, N, Arrange, p, q, num_warps, warp_id, BPB, rwork);
            }
        } else {
            static_assert(NT % 32 == 0, "logic requires an even number of warps");
            // Treat M as effectively larger due to asymmetries in the implementation
            constexpr auto raw_pq = pq_selector(2.5*M, N, NT);
            constexpr int p = raw_pq.p > 32u ? 32 : raw_pq.p;
            constexpr int q = NT / p;

            T rwork[M*N + 2*M];
            constexpr bool big_size = use_cta_big_per_batch(type_to_enum<T>, M, N, NT, BPB, Arch);
            cta_driver<T, big_size>(A, lda, tau_left, tau_right, tid, M, N, Arrange, p, q, BPB, rwork, swork);
        }
    }

    template<class T, unsigned M, unsigned N, arrangement Arrange>
    __device__ void thread_execute(T* A, int lda, T* tau_left, T* tau_right) {
        static_assert(M >= N, "gebrd doesn't support M < N");

        T rwork[M*N];
        thread_driver<T>(A, lda, tau_left, tau_right, 0, M, N, Arrange, 1, 1, rwork);
    }
} // namespace cusolverdx::detail::gebrd

#endif // CUSOLVERDX_DATABASE_GEBRD_CUH
