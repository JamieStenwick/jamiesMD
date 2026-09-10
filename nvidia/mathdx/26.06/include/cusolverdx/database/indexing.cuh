// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_INDEXING_CUH
#define CUSOLVERDX_DATABASE_INDEXING_CUH

#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx {
    namespace detail {

        template<class T>
        __device__ inline T& index_impl(T* A, unsigned lda, int i, int j, const bool transpose, const arrangement Arrange) {
            if (!transpose == (Arrange == col_major)) {
                return A[i + j * lda];
            } else {
                return A[i * lda + j];
            }
        }
        template<class T>
        __device__ inline const T& index_impl(const T* A, unsigned lda, int i, int j, const bool transpose, const arrangement Arrange) {
            if (!transpose == (Arrange == col_major)) {
                return A[i + j * lda];
            } else {
                return A[i * lda + j];
            }
        }

        template<class T>
        __device__ inline T& index(T* A, unsigned lda, int i, int j, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, false, Arrange);
        }

        template<class T>
        __device__ inline T& index(T* A, unsigned lda, int i, int j, const bool transpose, const arrangement Arrange) {
            return index_impl<T>(A, lda, i, j, transpose, Arrange);
        }

    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DATABASE_INDEXING_CUH
