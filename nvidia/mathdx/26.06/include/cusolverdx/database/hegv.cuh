// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_HEGV_CUH
#define CUSOLVERDX_DATABASE_HEGV_CUH

#include <cuComplex.h>

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/detail/type_enum.hpp"

#include "cusolverdx/database/util.cuh"
#include "cusolverdx/database/indexing.cuh"
#include "cusolverdx/database/potrf.cuh"
#include "cusolverdx/database/hegst.cuh"
#include "cusolverdx/database/heev.cuh"
#include "cusolverdx/database/trmm.cuh"
#include "cusolverdx/database/trsm.cuh"
#include "cusolverdx/database/hegv_db.cuh"

namespace cusolverdx::detail::hegv {

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned N, job Job, int Arch) {
        auto thresholds = suggested_bpb_size_thresholds(T, Job != job::no_vectors, Arch);
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

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned N, job Job, unsigned BPB, int Arch) {
        if (BPB > 1u) {
            return ((BPB + 32 - 1) / 32) * 32;
        } else {
            auto thresholds = suggested_block_dim_size_thresholds(T, Job != job::no_vectors, Arch);
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

    constexpr inline __device__ __host__ int workspace_size(type_enum T, unsigned N, int itype, job jobV, unsigned NT, unsigned BPB, int Arch) {
        return BPB
               + const_max(hegst::workspace_size(T, N, itype, NT, BPB, Arch),
                           heev::workspace_size(T, N, jobV, NT, BPB, Arch));
    }

    constexpr inline __device__ __host__ int workspace_size_thread(type_enum T, unsigned N, int itype, job jobV) {
        return const_max(hegst::workspace_size_thread(T, N, itype),
                         heev::workspace_size_thread(T, N, jobV));
    }

    template<class T, unsigned N, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, int itype, job jobV, unsigned NT, unsigned BPB, int Arch, class real_t = real_type_t<T>>
    __device__ void block_execute(T* A, int lda, T* B, int ldb, real_t* lambda, T* workspace, int* info, unsigned tid) {

        static_assert(itype == 1 || itype == 2 || itype == 3, "EigType for hegv must be 1, 2, or 3");
        static_assert(jobV == job::no_vectors || jobV == job::overwrite_vectors, "HEGV only supports job::no_vectors and job::overwrite_vectors.");

        static_assert(sizeof(int) <= sizeof(T), "HEGV workspace assumes a larger datatype");
        int* const potrf_info = reinterpret_cast<int*>(workspace);
        workspace += BPB;

        potrf::block_execute<T, N, Fill, ArrangeB, NT, BPB, Arch>(B, ldb, potrf_info, tid);
        __syncthreads();

        // If all Cholesky's failed, can return early
        // TODO figure out how to partially short circuit if cholesky failed
        bool any_success = false;
        for (int i = 0; i < int(BPB); ++i) {
            // TODO distribute this search across threads
            if (potrf_info[i] == 0) {
                any_success = true;
            }
        }
        if (!any_success) {
            for (int i = tid; i < int(BPB); i += NT) {
                info[i] = potrf_info[i] + N;
            }
            return;
        }

        hegst::block_execute<T, N, Fill, ArrangeA, ArrangeB, itype, NT, BPB, Arch>(A, lda, B, ldb, workspace, tid);
        __syncthreads();

        heev::block_execute<T, N, Fill, ArrangeA, jobV, NT, BPB, Arch>(A, lda, lambda, workspace, info, tid);
        __syncthreads();

        for (unsigned b = tid; b < BPB; b += NT) {
            const int p = potrf_info[b];
            const int e = info[b];
            info[b]     = (p != 0) ? N+p : e;
        }

        if constexpr (jobV == job::overwrite_vectors) {
            if constexpr (itype == 1 || itype == 2) {
                constexpr auto trans = Fill == fill_mode::lower ? transpose::conj_transposed : transpose::non_transposed;
                trsm::block_execute<T, N, N, side::left, diag::non_unit, trans, Fill, ArrangeB, ArrangeA, NT, BPB>(
                    B, ldb, A, lda, tid);
            } else {
                constexpr auto trans = Fill == fill_mode::lower ? transpose::non_transposed : transpose::conj_transposed;
                trmm::block_execute<T, N, N, side::left, diag::non_unit, trans, Fill, ArrangeB, ArrangeA, NT, BPB>(
                    B, ldb, A, lda, tid);
            }
            __syncthreads();
        }
    }

    template<class T, unsigned N, fill_mode Fill, arrangement ArrangeA, arrangement ArrangeB, int itype, job jobV, class real_t = real_type_t<T>>
    __device__ void thread_execute(T* A, int lda, T* B, int ldb, real_t* lambda, T* workspace, int* info) {

        static_assert(itype == 1 || itype == 2 || itype == 3, "EigType for hegv must be 1, 2, or 3");
        static_assert(jobV == job::no_vectors || jobV == job::overwrite_vectors, "HEGV only supports job::no_vectors and job::overwrite_vectors.");

        potrf::thread_execute<T, N, Fill, ArrangeB>(B, ldb, info);
        if (*info != 0) {
            *info += N; // adjust to correct range
            return;
        }

        hegst::thread_execute<T, N, Fill, ArrangeA, ArrangeB, itype>(A, lda, B, ldb, workspace);
        heev::thread_execute<T, N, Fill, ArrangeA, jobV>(A, lda, lambda, workspace, info);

        if constexpr (jobV == job::overwrite_vectors) {
            if constexpr (itype == 1 || itype == 2) {
                constexpr auto trans = Fill == fill_mode::lower ? transpose::conj_transposed : transpose::non_transposed;
                trsm::thread_execute<T, N, N, side::left, diag::non_unit, trans, Fill, ArrangeB, ArrangeA>(B, ldb, A, lda);
            } else {
                constexpr auto trans = Fill == fill_mode::lower ? transpose::non_transposed : transpose::conj_transposed;
                trmm::thread_execute<T, N, N, side::left, diag::non_unit, trans, Fill, ArrangeB, ArrangeA>(B, ldb, A, lda);
            }
        }
    }

} // namespace cusolverdx::detail::hegv

#endif // CUSOLVERDX_DATABASE_HEGV_CUH
