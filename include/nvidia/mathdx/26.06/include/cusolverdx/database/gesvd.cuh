// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_GESVD_CUH
#define CUSOLVERDX_DATABASE_GESVD_CUH

#include "cusolverdx/operators/enums.hpp"

#include "cusolverdx/detail/type_enum.hpp"

#include "cusolverdx/detail/block_execution_bugs.hpp"
#include "cusolverdx/database/gebrd.cuh"
#include "cusolverdx/database/bdsvd.cuh"
#include "cusolverdx/database/ungqr.cuh"
#include "cusolverdx/database/unglq.cuh"
#include "cusolverdx/database/util.cuh"

// using repack_tridiag from heev
#include "cusolverdx/database/heev.cuh"
#include "cusolverdx/database/gesvd_db.cuh"

namespace cusolverdx::detail::gesvd {

    template<class T>
    __device__ void ungbr_setup_U(T* A, const int lda, const int A_batch_size, T* U, const int ldu, const int U_batch_size, const job jobU, const unsigned tid,
                                  const unsigned M, const unsigned N, const arrangement ArrangeA, const arrangement ArrangeU, const int NT, const int BPB);
    template<class T>
    __device__ void ungbr_setup_VT(T* A, const int lda, const int A_batch_size, T* VT, const int ldvt, const int VT_batch_size, const job jobVT, const unsigned tid,
                                   const unsigned M, const unsigned N, const arrangement ArrangeA, const arrangement ArrangeVT, const int NT, const int BPB);

    constexpr inline __device__ __host__ unsigned suggested_batches(type_enum T, unsigned M, unsigned N, job JobU, job JobVT, int Arch) {
        // TODO tune for gesvd
        auto thresholds = gesvd::suggested_bpb_size_thresholds(T, M, N, JobU, JobVT, Arch);
        if (M*N <= thresholds[0] * thresholds[0]) {
            return 32;
        } else if (M*N <= thresholds[1] * thresholds[1]) {
            return 16;
        } else if (M*N <= thresholds[2] * thresholds[2]) {
            return 8;
        } else if (M*N <= thresholds[3] * thresholds[3]) {
            return 4;
        } else if (M*N <= thresholds[4] * thresholds[4]) {
            return 2;
        } else {
            return 1;
        }
    }

    constexpr inline __device__ __host__ dim3 suggested_block_dim(type_enum T, unsigned M, unsigned N, job JobU, job JobVT, unsigned BPB, int Arch) {
        // TODO tune for gesvd
        if (BPB > 1) {
            // TODO review this choice
            return ((BPB+32-1)/32)*32;
        } else {
            bool compute_vectors = JobU != job::no_vectors || JobVT != job::no_vectors;
            auto thresholds = gesvd::suggested_block_dim_size_thresholds(T, compute_vectors, Arch);
            if (M*N <= thresholds[0] * thresholds[0]) {
                return 32;
            } else if (M*N <= thresholds[1] * thresholds[1]) {
                return 64;
            } else if (M*N <= thresholds[2] * thresholds[2]) {
                return 128;
            } else if (M*N <= thresholds[3] * thresholds[3]) {
                return 256;
            } else {
                return 512;
            }
        }
    }

    // Number of words of type T needed as workspace
    constexpr inline __device__ __host__ int workspace_size(type_enum T, unsigned M, unsigned N, job jobU, job jobVT, unsigned NT, unsigned BPB, int Arch) {
        // Each batch needs 2*N words for tau_left and tau_right (reused for e), plus gebrd needs a little extra workspace in some cases

        int tau_size = 2*N*BPB;
        int gebrd_workspace = gebrd::workspace_size(T, M, N, NT, BPB, Arch);
        int e_size = (N-1) * BPB;

        bool compute_vectors = (jobU != job::no_vectors) || (jobVT != job::no_vectors);

        return compute_vectors ? tau_size + const_max(gebrd_workspace, e_size)
                               : const_max(tau_size + gebrd_workspace, e_size);
    }

    constexpr inline __device__ __host__ int workspace_size_thread([[maybe_unused]] type_enum T, [[maybe_unused]] unsigned M, unsigned N, job jobU, job jobVT) {
        // Each batch needs 2*N words for tau_left and tau_right (reused for e)

        int tau_size = 2*N;
        int e_size = (N-1);

        bool compute_vectors = (jobU != job::no_vectors) || (jobVT != job::no_vectors);

        return compute_vectors ? tau_size + e_size
                               : const_max(tau_size, e_size);
    }


    // TODO consider adding initial QR factorization for tall-skinny matrices
    // Based on conventional FLOP counts, theoretical crossover point for initial QR factorization is
    // * no vectors: m \geq 5/3 n
    // * vectors: m \geq 16/9 n
    // For Dx, I'm guessing matrices need to be even more skewed for it to be worthwhile
    // Initial implementation can probably start with just gebrd

    // TODO go through an rename "VT" to "VH" everywhere

    template<class T, unsigned M, unsigned N, job jobU, job jobVT, arrangement ArrangeA, arrangement ArrangeU, arrangement ArrangeVT, unsigned NT, unsigned BPB, int Arch, class real_t = real_type_t<T>>
    __device__ void block_execute(T* A, int lda, real_type_t<T>* sigma, T* U, int ldu, T* VT, int ldvt, T* workspace, int* info, unsigned tid) {

        static_assert(jobU != job::overwrite_vectors || jobVT != job::overwrite_vectors, "Cannot overwrite A array with both U and VT.");
        static_assert(jobU != job::multiply_vectors && jobVT != job::multiply_vectors, "GESVD doesn't support job::multiply_vectors.");

        CUSOLVERDX_NVBUG_5908003_GESVD_HANDLER(Arch, is_real_v<T>, jobU, jobVT, M, N);

        // Implementation exploits the assumption that M >= N
        // N.B., svdvals(A) = svdvals(A^T) for both real and complex
        // N.B., A = U S VT  <=>  A^T = VT^T S U^T for both real and complex
        if constexpr (M < N) {
            constexpr auto newArrangeA = ArrangeA == col_major ? row_major : col_major;
            constexpr auto newArrangeU = ArrangeU == col_major ? row_major : col_major;
            constexpr auto newArrangeVT = ArrangeVT == col_major ? row_major : col_major;
            return block_execute<T, N, M, jobVT, jobU, newArrangeA, newArrangeVT, newArrangeU, NT, BPB, Arch, real_t>(
                    A, lda, sigma, VT, ldvt, U, ldu, workspace, info, tid);
        } else {

            // Reduce to bidiagonal
            T* tau_left = workspace;
            T* tau_right = tau_left + N*BPB;
            T* swork = tau_right + N*BPB;

            gebrd::block_execute<T, M, N, ArrangeA, NT, BPB, Arch>(A, lda, tau_left, tau_right, tid, swork);

            // Repack bidiagonal matrix
            __syncthreads();
            real_t* d = sigma;

            real_t* e;
            // If no vectors are requested, tau's are not needed
            if (jobU == job::no_vectors && jobVT == job::no_vectors) {
                e = reinterpret_cast<real_t*>(workspace);
            } else {
                e = reinterpret_cast<real_t*>(swork);
            }
            const int A_batch_stride = (ArrangeA == col_major) ? lda*N : M*lda;
            heev::repack_tridiag(A, lda, A_batch_stride, d, e, tid, N, fill_mode::upper, ArrangeA, NT, BPB);
            __syncthreads();


            // setup U and VT pointers
            constexpr bool overwriteU  = jobU  == job::overwrite_vectors;
            constexpr bool overwriteVT = jobVT == job::overwrite_vectors;
            T*                    act_U         = overwriteU  ? A : U;
            int                   act_ldu       = overwriteU  ? lda : ldu;
            constexpr arrangement act_ArrangeU  = overwriteU  ? ArrangeA : ArrangeU;
            T*                    act_VT        = overwriteVT ? A : VT;
            int                   act_ldvt      = overwriteVT ? lda : ldvt;
            constexpr arrangement act_ArrangeVT = overwriteVT ? ArrangeA : ArrangeVT;

            const int U_batch_size = overwriteU ? A_batch_stride : (jobU == job::all_vectors) ? act_ldu * M : (ArrangeU == col_major ? act_ldu * N : M * act_ldu);
            const int VT_batch_size = overwriteVT ? A_batch_stride : act_ldvt * N;

            // Generate U and V from the bidiagonalization step
            // NB., setup for U is done first since setup for VT requires setting a row and column
            if (jobU != job::no_vectors) {
                ungbr_setup_U(A, lda, A_batch_stride, U, ldu, U_batch_size, jobU, tid, M, N, ArrangeA, ArrangeU, NT, BPB);
                __syncthreads();
            }
            // Note here using act_Vt, a pointer either VT or A
            if (jobVT != job::no_vectors) {
                ungbr_setup_VT(A, lda, A_batch_stride, act_VT, act_ldvt, VT_batch_size, jobVT, tid, M, N, ArrangeA, act_ArrangeVT, NT, BPB);
                __syncthreads();
            }

            if (jobU == job::all_vectors) {
                ungqr::block_execute<T, M, M, N, act_ArrangeU, NT, BPB>(act_U, act_ldu, U_batch_size, tau_left, N, tid);
            } else if (jobU == job::overwrite_vectors || jobU == job::some_vectors) {
                ungqr::block_execute<T, M, N, N, act_ArrangeU, NT, BPB>(act_U, act_ldu, U_batch_size, tau_left, N, tid);
            }
            if (jobVT != job::no_vectors) {
                unglq::block_execute<T, N-1, N-1, N-1, act_ArrangeVT, NT, BPB>(act_VT + (1 + act_ldvt), act_ldvt, VT_batch_size, tau_right, N, tid);
            }
            __syncthreads();

            
            constexpr int M_U = jobU == job::no_vectors ? 0 : M;
            constexpr int M_V = jobVT == job::no_vectors ? 0 : N;


            bdsvd::block_execute<real_t, T, N, M_U, M_V, false, false, act_ArrangeU, act_ArrangeVT, NT, BPB, Arch>(
                d, e, act_U, act_ldu, U_batch_size, act_VT, act_ldvt, VT_batch_size, info, tid);
                
            }
        }

    template<class T, unsigned M, unsigned N, job jobU, job jobVT, arrangement ArrangeA, arrangement ArrangeU, arrangement ArrangeVT, class real_t = real_type_t<T>>
    __device__ void thread_execute(T* A, int lda, real_type_t<T>* sigma, T* U, int ldu, T* VT, int ldvt, T* workspace, int* info) {

        static_assert(jobU != job::overwrite_vectors || jobVT != job::overwrite_vectors, "Cannot overwrite A array with both U and VT.");
        static_assert(jobU != job::multiply_vectors && jobVT != job::multiply_vectors, "GESVD doesn't support job::multiply_vectors.");

        // Implementation exploits the assumption that M >= N
        // N.B., svdvals(A) = svdvals(A^T) for both real and complex
        // N.B., A = U S VT  <=>  A^T = VT^T S U^T for both real and complex
        if constexpr (M < N) {
            constexpr auto newArrangeA = ArrangeA == col_major ? row_major : col_major;
            constexpr auto newArrangeU = ArrangeU == col_major ? row_major : col_major;
            constexpr auto newArrangeVT = ArrangeVT == col_major ? row_major : col_major;
            return thread_execute<T, N, M, jobVT, jobU, newArrangeA, newArrangeVT, newArrangeU, real_t>(
                    A, lda, sigma, VT, ldvt, U, ldu, workspace, info);
        } else {

            // Reduce to bidiagonal
            T* tau_left = workspace;
            T* tau_right = tau_left + N;
            T* swork = tau_right + N;

            gebrd::thread_execute<T, M, N, ArrangeA>(A, lda, tau_left, tau_right);

            // Repack bidiagonal matrix
            real_t* d = sigma;
            real_t* e;
            // If no vectors are requested, tau's are not needed
            if (jobU == job::no_vectors && jobVT == job::no_vectors) {
                e = reinterpret_cast<real_t*>(workspace);
            } else {
                e = reinterpret_cast<real_t*>(swork);
            }
            // One thread processes one batch, so batch stride is 0
            heev::repack_tridiag(A, lda, 0, d, e, 0, N, fill_mode::upper, ArrangeA, 1, 1);

            // setup U and VT pointers
            constexpr bool overwriteU  = jobU  == job::overwrite_vectors;
            constexpr bool overwriteVT = jobVT == job::overwrite_vectors;
            T*                    act_U         = overwriteU  ? A : U;
            int                   act_ldu       = overwriteU  ? lda : ldu;
            constexpr arrangement act_ArrangeU  = overwriteU  ? ArrangeA : ArrangeU;
            T*                    act_VT        = overwriteVT ? A : VT;
            int                   act_ldvt      = overwriteVT ? lda : ldvt;
            constexpr arrangement act_ArrangeVT = overwriteVT ? ArrangeA : ArrangeVT;

            // Generate U and V from the bidiagonalization step
            // NB., setup for U is done first since setup for VT requires setting a row and column

            if (jobU != job::no_vectors) {
                ungbr_setup_U(A, lda, 0, U, ldu, 0, jobU, 0, M, N, ArrangeA, ArrangeU, 1, 1);
            }
            // Note here using act_Vt, a pointer either VT or A
            if (jobVT != job::no_vectors) {
                ungbr_setup_VT(A, lda, 0, act_VT, act_ldvt, 0, jobVT, 0, M, N, ArrangeA, act_ArrangeVT, 1, 1);
            }

            if (jobU == job::all_vectors) {
                ungqr::thread_execute<T, M, M, N, act_ArrangeU>(act_U, act_ldu, tau_left);
            } else if (jobU == job::overwrite_vectors || jobU == job::some_vectors) {
                ungqr::thread_execute<T, M, N, N, act_ArrangeU>(act_U, act_ldu, tau_left);
            }
            if (jobVT != job::no_vectors) {
                unglq::thread_execute<T, N-1, N-1, N-1, act_ArrangeVT>(act_VT + (1 + act_ldvt), act_ldvt, tau_right);
            }


            constexpr int M_U = jobU == job::no_vectors ? 0 : M;
            constexpr int M_V = jobVT == job::no_vectors ? 0 : N;

            // Solve for values
            bdsvd::thread_execute<real_t, T, N, M_U, M_V, false, false, act_ArrangeU, act_ArrangeVT>(d, e, act_U, act_ldu, 0, act_VT, act_ldvt, 0, info);

        }
    }

} // namespace cusolverdx::detail::gesvd

#endif // CUSOLVERDX_DATABASE_GESVD_CUH
