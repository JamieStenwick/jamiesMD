// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_UNGQR_DB_CUH
#define CUSOLVERDX_DATABASE_UNGQR_DB_CUH

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx {
    namespace detail {
        namespace ungqr {

            // Size thresholds for suggesting batches per block
            constexpr inline __device__ __host__ threshold_array<5> suggested_bpb_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
                if (T == type_enum::real_f32) {
                    return {3 * 3, 6 * 6, 7 * 7, 12 * 12, 19 * 19};
                } else if (T == type_enum::real_f64) {
                    return {3 * 3, 3 * 3, 4 * 4, 12 * 12, 15 * 15};
                } else if (T == type_enum::complex_f32) {
                    return {3 * 3, 3 * 3, 4 * 4, 12 * 12, 15 * 15};
                } else { // complex_f64
                    return {3 * 3, 3 * 3, 3 * 3, 7 * 7, 11 * 11};
                }
            }

            constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned M, unsigned N, int Arch) {
                auto thresholds = suggested_bpb_size_thresholds(T, Arch);
                if (M * N <= thresholds[0]) {
                    return 32;
                } else if (M * N <= thresholds[1]) {
                    return 16;
                } else if (M * N <= thresholds[2]) {
                    return 8;
                } else if (M * N <= thresholds[3]) {
                    return 4;
                } else if (M * N <= thresholds[4]) {
                    return 2;
                } else {
                    return 1;
                }
            }

            // Size thresholds for suggesting the block dim for a single batch
            constexpr inline __device__ __host__ threshold_array<4> suggested_block_dim_size_thresholds(type_enum T, [[maybe_unused]] int Arch) {
                // Tuned for H100. For float and double real, tune until 128, and complex<float> and complex<double> until 96
                constexpr unsigned int INF = unsigned(-1);
                if (T == type_enum::real_f32) {
                    return {31 * 31, 71 * 71, 95 * 95, INF};
                } else if (T == type_enum::real_f64) {
                    return {32 * 32, 64 * 64, 123 * 123, INF};
                } else if (T == type_enum::complex_f32) {
                    return {31 * 31, 63 * 63, INF, INF};
                } else { // complex_f64
                    return {31 * 31, 63 * 63, 91 * 91, INF};
                }
            }

            constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned M, unsigned N, unsigned BPB, int Arch) {
                // Targets throughput bound cases

                if (BPB > 1) {
                    if (M * N <= 28u * 28u) {
                        return BPB <= 32u ? 32 : 64;
                    } else if (M * N <= (T == type_enum::real_f32 ? 64u * 64u : 32u * 32u)) {
                        if (BPB == 2) {
                            return 64;
                        } else {
                            // Use 3 or 4 warps, depending on what results in fewer idle warps for the last wave of batches
                            unsigned rem_3 = (BPB % 3 == 0) ? 0 : 3 - (BPB % 3);
                            unsigned rem_4 = (BPB % 4 == 0) ? 0 : 4 - (BPB % 4);
                            return rem_3 < rem_4 ? 96 : 128;
                        }
                    } else {
                        // For large sizes, just use suggestion for 1 batch per block.
                        return suggested_block_dim(T, M, N, 1, Arch);
                    }
                } else {
                    auto thresholds = suggested_block_dim_size_thresholds(T, Arch);
                    if (M * N <= thresholds[0]) {
                        return 32;
                    } else if (M * N <= thresholds[1]) {
                        return 64;
                    } else if (M * N <= thresholds[2]) {
                        return 128;
                    } else if (M * N <= thresholds[3]) {
                        return 256;
                    } else {
                        return 512;
                    }
                }
            }

        } // namespace ungqr
    }  // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_UNGQR_DB_CUH
