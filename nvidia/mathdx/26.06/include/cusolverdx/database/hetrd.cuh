// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HETRD_CUH
#define CUSOLVERDX_DATABASE_HETRD_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/detail/type_enum.hpp"

#include "cusolverdx/database/hetrd_db.cuh"
#include "cusolverdx/database/grid_utils.cuh"

namespace cusolverdx::detail::hetrd {
    template<class T>
    __device__ void thread_driver(T* A, int lda, T* tau, unsigned tid, unsigned N, const fill_mode Fill, const arrangement Arrange, unsigned NT, unsigned BPB, T* rwork);

    template<class T>
    __device__ void warp_driver(T* A, int lda, T* tau, unsigned tid, unsigned subwarp_size, unsigned N, const fill_mode Fill, const arrangement Arrange, unsigned NT, unsigned p, unsigned q, unsigned BPB, T* rwork);

    template<class T, bool big_size>
    __device__ void cta_driver(T* A, int lda, T* tau, T* swork, unsigned tid, unsigned N, const fill_mode Fill, const arrangement Arrange, unsigned NT, unsigned p, unsigned q, unsigned BPB, T* rwork, T* rwork2);

    // hetrd makes some different choices from the general selector in grid_utils.cuh
    constexpr inline pq_pair __host__ __device__ hetrd_pq_selector(unsigned NT) {
        // Sizes for warp_impl
        if (NT == 32) {
            return {8, 4};
        } else if (NT == 16) {
            return {4, 4};
        } else if (NT == 8) {
            return {4, 2};
        } else if (NT == 4) {
            return {2, 2};
        } else if (NT == 2) {
            return {2, 1};
        } else if (NT == 1) {
            return {1, 1};
        }

        // Sizes for cta_impl
        if (NT == 64) {
            return {8, 8};
        } else if (NT == 96) {
            return {8, 12};
        } else if (NT < 16u*32u) { // 16*8 <= NT < 16*32
            return {16, NT / 16};
        } else { // 32*16 <= NT <= 32*32
            return {32, NT / 32};
        }
    }

    constexpr inline __device__ __host__ bool use_thread_per_batch(type_enum T, unsigned N, unsigned NT, unsigned BPB, int Arch) {
        return N <= tiny_threshold(T, Arch) || NT % 32 != 0 || BPB > NT / 2;
    }

    constexpr inline __device__ __host__ bool use_warp_per_batch([[maybe_unused]] type_enum T, [[maybe_unused]] unsigned N, unsigned NT, unsigned BPB, [[maybe_unused]] int Arch) {
        return BPB >= (NT / 32) && NT % 32 == 0;
    }

    constexpr inline __device__ __host__ bool use_cta_big_per_batch(type_enum T, unsigned N, [[maybe_unused]] unsigned NT, [[maybe_unused]] unsigned BPB, int Arch) {
        return N >= big_threshold(T, Arch);
    }

    // Number of words of type T needed as workspace
    constexpr inline __device__ __host__ int workspace_size(type_enum T, unsigned N, unsigned NT, unsigned BPB, int Arch) {
        if (use_thread_per_batch(T, N, NT, BPB, Arch) || use_warp_per_batch(T, N, NT, BPB, Arch)) {
            return 0;
        } else { // cta impl
            const auto [p, q] = hetrd_pq_selector(NT);
            int num_warps = NT / 32;
            return ((N + p - 1)/p) * p * (1 + num_warps);
        }
    }

    // Batches of A have the normal stride for an NxN matrix of Arrange with lda leading dimension
    // Batches of tau have a stride of N-1
    template<class T, unsigned N, fill_mode Fill, arrangement Arrange, unsigned NT, unsigned BPB, int Arch>
    __device__ void block_execute(T* A, int lda, T* tau, T* swork, unsigned tid) {
        if constexpr (use_thread_per_batch(type_to_enum<T>, N, NT, BPB, Arch)) {
            T rwork[N];
            thread_driver<T>(A, lda, tau, tid, N, Fill, Arrange, NT, BPB, rwork);
        } else if constexpr (use_warp_per_batch(type_to_enum<T>, N, NT, BPB, Arch)) {
            // Using warp or partial warp driver

            constexpr unsigned subwarp_size = compute_subwarp_size<BPB, NT>();
            static_assert(32 % subwarp_size == 0, "The subwarp size must be a power of 2");
            const auto [p, q] = hetrd_pq_selector(subwarp_size);

            T rwork[N*N + 4*N];
            warp_driver<T>(A, lda, tau, tid, subwarp_size, N, Fill, Arrange, NT, p, q, BPB, rwork);
        } else {
            static_assert(NT % 32 == 0, "The CTA implementation requires NT to be a multiple of 32");
            const auto [p, q] = hetrd_pq_selector(NT);


            constexpr bool big_size = use_cta_big_per_batch(type_to_enum<T>, N, NT, BPB, Arch);
            T rwork[N*N];
            T rwork2[4*N];
            cta_driver<T, big_size>(A, lda, tau, swork, tid, N, Fill, Arrange, NT, p, q, BPB, rwork, rwork2);
        }
    }

    template<class T, unsigned N, fill_mode Fill, arrangement Arrange>
    __device__ void thread_execute(T* A, int lda, T* tau) {
        T rwork[N];
        thread_driver<T>(A, lda, tau, 0, N, Fill, Arrange, 1, 1, rwork);
    }
} // namespace cusolverdx::detail::hetrd

#endif // CUSOLVERDX_DATABASE_HETRD_CUH
