// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GESVD_DB_CUH
#define CUSOLVERDX_DATABASE_GESVD_DB_CUH

#include "cusolverdx/operators/enums.hpp"
#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail::gesvd {

    // Size thresholds for suggesting batches per block
    constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, unsigned M, unsigned N, job JobU, job JobVT, [[maybe_unused]] int Arch) {
        [[maybe_unused]] bool compute_no_vectors = JobU == job::no_vectors && JobVT == job::no_vectors;
        [[maybe_unused]] bool compute_one_vectors = (JobU != job::no_vectors && JobVT == job::no_vectors) || (JobU == job::no_vectors && JobVT != job::no_vectors);
        bool compute_two_vectors = JobU != job::no_vectors && JobVT != job::no_vectors;
        unsigned mn_diff = (M > N) ? (M - N) : (N - M);
        bool all_vectors_with_large_M_N_diff =
            (JobU == job::all_vectors || JobVT == job::all_vectors) && mn_diff >= 10u;

        if (T == type_enum::real_f32) {
            if (compute_two_vectors && !all_vectors_with_large_M_N_diff) {
                // TODO tune for this case
                return {4, 15, 15, 18, 18};
            } else if (compute_one_vectors && !all_vectors_with_large_M_N_diff) {
                // TODO tune for this case
                return {4, 15, 15, 18, 18};
            } else if (compute_no_vectors) {
                return {4, 11, 19, 24, 31};
            } else { // either or both could be all_vectors with possible large U/T size
                return {4, 6, 8, 8, 8};
            }
        } else if (T == type_enum::real_f64) {
            if (compute_two_vectors && !all_vectors_with_large_M_N_diff) {
                return {4, 15, 15, 17, 20};
            } else if (compute_one_vectors && !all_vectors_with_large_M_N_diff) {
                return {4, 15, 15, 17, 20};
            } else if (compute_no_vectors) {
                return {4, 11, 16, 23, 32};
            } else {
                return {4, 4, 8, 8, 8};
            }
        } else if (T == type_enum::complex_f32) { //complex<float>
            if (compute_two_vectors && !all_vectors_with_large_M_N_diff) {
                return {4, 14, 15, 15, 18};
            } else if (compute_one_vectors && !all_vectors_with_large_M_N_diff) {
                return {4, 14, 15, 15, 18};
            } else if (compute_no_vectors) {
                return {4, 10, 10, 16, 27};
            } else {
                return {4, 4, 8, 8, 8};
            }
        } else { // complex<double>
            if (compute_two_vectors && !all_vectors_with_large_M_N_diff) {
                return {4, 11, 15, 15, 16};
            } else if (compute_one_vectors && !all_vectors_with_large_M_N_diff) {
                return {4, 11, 15, 15, 16};
            } else if (compute_no_vectors) {
                return {4, 10, 10, 23, 32};
            } else {
                return {4, 4, 8, 8, 8};
            }
        }
    }

    // Size thresholds for suggesting the block dim for a single batch
    constexpr inline __device__ __host__ threshold_array<4> suggested_block_dim_size_thresholds(type_enum T, bool compute_vectors, [[maybe_unused]] int Arch) {
        // Experimentally tuned for H100-PCIe
        if (T == type_enum::real_f32) {
            // tuned for N=1:128:4
            if (compute_vectors) {
                return {28, 32, 64, 88};
            } else {
                return {48, 52, 64, 112};
            }
        } else if (T == type_enum::real_f64) {
            // tuned for N=1:128:4
            if (compute_vectors) {
                return {28, 36, 64, 80};
            } else {
                return {44, 64, 92, 112};
            }
        } else if (T == type_enum::complex_f32) { //complex<float>
            // tuned for N=1:96:4
            if (compute_vectors) {
                return {28, 32, 48, 80};
            } else {
                return {36, 44, 48, 80};
            }
        } else { // complex<double>
            // tuned for N=1:96:4
            if (compute_vectors) {
                return {28, 36, 48, 200};
            } else {
                return {36, 52, 68, 80};
            }
        }
    }

} // namespace cusolverdx::detail::gesvd

#endif // CUSOLVERDX_DATABASE_GESVD_DB_CUH
