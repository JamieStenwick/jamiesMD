// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GRID_UTILS_CUH
#define CUSOLVERDX_DATABASE_GRID_UTILS_CUH

#include "cusolverdx/database/util.cuh"

namespace cusolverdx {
    namespace detail {

        struct pq_pair {
            unsigned p, q;
        };

        // Selects p and q for a given problem dimension
        // Because this function is constexpr, it should be evaluated at compile time
        constexpr __device__ __host__ pq_pair pq_selector(unsigned M, unsigned N, unsigned NT) {
            // We manually specify some basic configurations for square matrices and "nice" NT
            if (M == N) {
                if (NT == 128) {
                    return {8, 16};
                } else if (NT == 96) {
                    return {8, 12};
                } else if (NT == 64) {
                    return {8, 8};
                } else if (NT == 32) {
                    return {4, 8};
                } else if (NT == 16) {
                    return {4, 4};
                } else if (NT == 8) {
                    return {2, 4};
                } else if (NT == 4) {
                    return {2, 2};
                } else if (NT == 2) {
                    return {1, 2};
                } else if (NT == 1) {
                    return {1, 1};
                }
                // Other NT fall through to the general logic
            }

            // The following logic assumes M > N.  If that's not the case, flip the args then undo it at the end
            bool flip = M < N;
            if (flip) {
                auto temp = M;
                M = N;
                N = temp;
            }

            unsigned p = 0, q = 0;
            if ((NT & (NT - 1)) == 0) {
                // if NT is a power of 2, find p, q that are also powers of 2

                // Find the smallest p such that p*q==NT and 2*p/q >= M/N
                // Because M > N and NT >= 2, the minimial p is at least 1
                p = NT;
                q = 1;
                float ideal_ratio = M / float(N);
                // Check p, q would still be valid after another step
                while (ideal_ratio < 2 * (p/2) / float(2*q)) {
                    p /= 2;
                    q *= 2;
                }
            } else {
                // general algorithm that won't provide as good of M, N

                // Find p, q such that p*q==NT and p/q is as close as possible to M/N

                // First, find largest p s.t., p*q==NT and p < NT*M/(M+N)
                unsigned p_lo = NT * M / (M + N);
                while (p_lo * (NT / p_lo) != NT) { // Loop always terminates because p_lo=1 is valid
                    --p_lo;
                }
                unsigned q_lo = NT / p_lo;

                // Next, find smallest p s.t., p*q==NT and p > NT*M/(M+N)
                unsigned p_hi = (NT * M - 1) / (M + N) + 1;
                while (p_hi * (NT / p_hi) != NT) { // Loop always terminates because p_hi=NT is valid
                    ++p_hi;
                }
                unsigned q_hi = NT / p_hi;

                // Determine whether p_lo/q_lo or p_hi/q_hi is closer
                double lo_fact = q_lo * M / double(p_lo * N);
                double hi_fact = p_hi * N / double(q_hi * M);

                // On ties, choose hi since M/N will increase during the factorization until M=N
                if (lo_fact >= hi_fact) {
                    p = p_hi;
                    q = q_hi;
                } else {
                    p = p_lo;
                    q = q_lo;
                }

            }

            return flip ? pq_pair{q, p} : pq_pair{p, q};
        }

        // Computes the number of threads to use per sub-warp so that the number of subwarps evenly divides BPB
        // Requires NT to be a multiple of 32 and assumes that the number of threads is larger than the number of batches
        template<unsigned BPB, unsigned NT>
        constexpr inline unsigned __device__ compute_subwarp_size() {
            static_assert(NT % 32 == 0, "warp-based implementation currently assumes NT is a multiple of 32");
            static_assert(BPB < NT, "warp-based implementation assumes that there are more threads than batches");

            constexpr unsigned num_warps = NT/32;
            constexpr unsigned max_B_per_warp = pow2_divisor(BPB);

            if (BPB <= num_warps) {
                return 32;
            } else {
                return 32 / max_B_per_warp;
            }
        }

    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_GRID_UTILS_CUH
