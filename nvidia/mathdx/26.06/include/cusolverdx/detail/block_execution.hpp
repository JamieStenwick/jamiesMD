// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_BLOCK_EXECUTION_HPP
#define CUSOLVERDX_DETAIL_BLOCK_EXECUTION_HPP

#include "cusolverdx/detail/solver_execution.hpp"

namespace cusolverdx {
    namespace detail {

        template<class... Operators>
        class block_execution: public solver_execution<Operators...> {
            using this_type = block_execution<Operators...>;
            using base_type = solver_execution<Operators...>;

            // Import precision type from base class
            using typename base_type::this_solver_precision;

            /// ---- Constraints
            static_assert(base_type::has_block, "Block operator is missing in the solver description");
            static_assert(!has_operator_v<operator_type::blocks_per_cluster, base_type>, "Solver for block execution can't contain BlocksPerCluster<> operator");
            static_assert(!has_operator_v<operator_type::tile_size, base_type>, "Solver for block execution can't contain TileSize<> operator");

        public:
            static constexpr auto m_size = base_type::m_size;
            static constexpr auto n_size = base_type::n_size;
            static constexpr auto k_size = base_type::k_size;
            static constexpr auto lda    = base_type::lda;
            static constexpr auto ldb    = base_type::ldb;
            static constexpr auto ldc    = base_type::ldc;

            static constexpr unsigned int batches_per_block = base_type::this_solver_batches_per_block_v;

            // Import value types from base class
            using typename base_type::a_cuda_data_type;
            using typename base_type::a_data_type;
            using typename base_type::a_precision;
            using typename base_type::b_cuda_data_type;
            using typename base_type::b_data_type;
            using typename base_type::b_precision;
            using typename base_type::x_cuda_data_type;
            using typename base_type::x_data_type;
            using typename base_type::x_precision;

            using typename base_type::status_type;

            static constexpr auto type          = base_type::this_solver_type_v;
            static constexpr auto a_arrangement = base_type::this_solver_arrangement_a;
            static constexpr auto b_arrangement = base_type::this_solver_arrangement_b;
            static constexpr auto c_arrangement = base_type::this_solver_arrangement_c;
            static constexpr auto transpose     = base_type::this_solver_transpose_v;
            static constexpr auto fill_mode     = base_type::this_solver_fill_mode_v;
            static constexpr auto side          = base_type::this_solver_side_v;
            static constexpr auto diag          = base_type::this_solver_diag_v;
            static constexpr auto job           = base_type::this_solver_jobu_v;
            static constexpr auto jobu          = base_type::this_solver_jobu_v;
            static constexpr auto jobvt         = base_type::this_solver_jobvt_v;
            static constexpr auto eig_type    = base_type::this_solver_eig_type_v;
            static constexpr auto sm            = base_type::this_sm_v;

            // Import parent class execute overloads
            using base_type::execute;

        private:
            static constexpr type_enum dispatch_type = type_to_enum<a_cuda_data_type>;

            __host__ __device__ __forceinline__ static constexpr unsigned int get_suggested_batches_per_block() {
                static_assert(base_type::is_complete_v, "Can't provide suggested batches per block, description is not complete");
                // Gaussian elimination
                if constexpr (function_of_v<this_type> == function::potrf) {
                    return potrf::suggested_batches(dispatch_type, m_size, sm);
                } else if constexpr (function_of_v<this_type> == function::potrs) {
                    return potrs::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::posv) {
                    return posv::suggested_batches(dispatch_type, m_size, sm);
                } else if constexpr (function_of_v<this_type> == function::getrf_no_pivot || function_of_v<this_type> == function::modified_lu) {
                    return getrf_no_pivot::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::getrs_no_pivot) {
                    return getrs_no_pivot::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::gesv_no_pivot) {
                    return gesv_no_pivot::suggested_batches(dispatch_type, m_size, sm);
                } else if constexpr (function_of_v<this_type> == function::getrf_partial_pivot) {
                    return getrf_partial_pivot::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::getrs_partial_pivot) {
                    return getrs_partial_pivot::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::gesv_partial_pivot) {
                    return gesv_partial_pivot::suggested_batches(dispatch_type, m_size, sm);
                } else if constexpr (function_of_v<this_type> == function::gtsv_no_pivot) {
                    return gtsv_no_pivot::suggested_batches(dispatch_type, m_size, k_size, sm);
                // QR
                } else if constexpr (function_of_v<this_type> == function::gels) {
                    return gels::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::geqrf) {
                    return geqrf::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::gelqf) {
                    return gelqf::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::unmqr) {
                    return unmqr::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::unmlq) {
                    return unmlq::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::ungqr) {
                    return ungqr::suggested_batches(dispatch_type, m_size, n_size, sm);
                } else if constexpr (function_of_v<this_type> == function::unglq) {
                    return unglq::suggested_batches(dispatch_type, m_size, n_size, sm);
                // Eigenvalue
                } else if constexpr (function_of_v<this_type> == function::htev) {
                    return htev::suggested_batches(dispatch_type, m_size, job, sm);
                } else if constexpr (function_of_v<this_type> == function::heev) {
                    return heev::suggested_batches(dispatch_type, m_size, job, sm);
                } else if constexpr (function_of_v<this_type> == function::hegst) {
                    return hegst::suggested_batches(dispatch_type, m_size, eig_type, sm);
                } else if constexpr (function_of_v<this_type> == function::hegv) {
                    return hegv::suggested_batches(dispatch_type, m_size, job, sm);
                // SVD
                } else if constexpr (function_of_v<this_type> == function::bdsvd) {
                    return bdsvd::suggested_batches(dispatch_type, m_size, jobu, jobvt, sm);
                } else if constexpr (function_of_v<this_type> == function::gesvd) {
                    return gesvd::suggested_batches(dispatch_type, m_size, n_size, jobu, jobvt, sm);
                // BLAS
                } else if constexpr (function_of_v<this_type> == function::trsm) {
                    return trsm::suggested_batches(dispatch_type, m_size, n_size, side, sm);
                } else {
                    return 1;
                }
            }

            __host__ __device__ __forceinline__ static constexpr dim3 get_suggested_block_dim() {
                static_assert(base_type::is_complete_v, "Can't provide suggested block dimensions, description is not complete");
                // Gaussian elimination
                if constexpr (function_of_v<this_type> == function::potrf) {
                    return potrf::suggested_block_dim(dispatch_type, m_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::potrs) {
                    return potrs::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::posv) {
                    return posv::suggested_block_dim(dispatch_type, m_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::getrf_no_pivot || function_of_v<this_type> == function::modified_lu) {
                    return getrf_no_pivot::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::getrs_no_pivot) {
                    return getrs_no_pivot::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::gesv_no_pivot) {
                    return gesv_no_pivot::suggested_block_dim(dispatch_type, m_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::getrf_partial_pivot) {
                    return getrf_partial_pivot::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::getrs_partial_pivot) {
                    return getrs_partial_pivot::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::gesv_partial_pivot) {
                    return gesv_partial_pivot::suggested_block_dim(dispatch_type, m_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::gtsv_no_pivot) {
                    return gtsv_no_pivot::suggested_block_dim(dispatch_type, m_size, k_size, batches_per_block, sm);
                // QR
                } else if constexpr (function_of_v<this_type> == function::gels) {
                    return gels::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::geqrf) {
                    return geqrf::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::gelqf) {
                    return gelqf::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::unmqr) {
                    return unmqr::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::unmlq) {
                    return unmlq::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::ungqr) {
                    return ungqr::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::unglq) {
                    return unglq::suggested_block_dim(dispatch_type, m_size, n_size, batches_per_block, sm);
                // Eigenvalue
                } else if constexpr (function_of_v<this_type> == function::htev) {
                    return htev::suggested_block_dim(dispatch_type, m_size, job, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::heev) {
                    return heev::suggested_block_dim(dispatch_type, m_size, job, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::hegst) {
                    return hegst::suggested_block_dim(dispatch_type, m_size, eig_type, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::hegv) {
                    return hegv::suggested_block_dim(dispatch_type, m_size, job, batches_per_block, sm);
                // SVD
                } else if constexpr (function_of_v<this_type> == function::bdsvd) {
                    return bdsvd::suggested_block_dim(dispatch_type, m_size, jobu, jobvt, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::gesvd) {
                    return gesvd::suggested_block_dim(dispatch_type, m_size, n_size, jobu, jobvt, batches_per_block, sm);
                // BLAS
                } else if constexpr (function_of_v<this_type> == function::trsm) {
                    return trsm::suggested_block_dim(dispatch_type, m_size, n_size, side, batches_per_block, sm);
                } else {
                    return 256;
                }
            }

            __host__ __device__ __forceinline__ static constexpr dim3 get_block_dim() {
                static_assert(base_type::is_complete_v, "Can't provide block dimensions, description is not complete");
                if constexpr (base_type::has_block_dim) {
                    return base_type::this_block_dim_v;
                }
                return get_suggested_block_dim();
            }

            __host__ __device__ __forceinline__ static constexpr int get_workspace_size() {
                if constexpr (function_of_v<this_type> == function::heev) {
                    return heev::workspace_size(dispatch_type, m_size, job, max_threads_per_block, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::hegst) {
                    return hegst::workspace_size(dispatch_type, m_size, eig_type, max_threads_per_block, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::hegv) {
                    return hegv::workspace_size(dispatch_type, m_size, eig_type, job, max_threads_per_block, batches_per_block, sm);
                } else if constexpr (function_of_v<this_type> == function::gesvd) {
                    return gesvd::workspace_size(dispatch_type, m_size, n_size, jobu, jobvt, max_threads_per_block, batches_per_block, sm);
                } else {
                    return 0;
                }
            }

            __device__ __forceinline__ unsigned get_thread_id() {
                const auto dim = get_block_dim();
                __builtin_assume(threadIdx.x < dim.x);
                __builtin_assume(threadIdx.y < dim.y);
                __builtin_assume(threadIdx.z < dim.z);

                return threadIdx.x + dim.x * (threadIdx.y + dim.y * threadIdx.z);
            }

        public:
            inline static constexpr unsigned int get_shared_memory_size() { return get_shared_memory_size(lda, ldb, ldc); }

            // support both compile-time and run-time leading dimensions
            inline static constexpr unsigned int get_shared_memory_size(const unsigned int runtime_lda, const unsigned int runtime_ldb = ldb, const unsigned int runtime_ldc = ldc) {
                static_assert(base_type::is_complete_v, "Can't calculate shared memory, description is not complete");

                const unsigned int size_a    = smem_align(sizeof(a_data_type) * runtime_lda * ((a_arrangement == arrangement::col_major) ? n_size : m_size));
                const unsigned int size_b    = smem_align(sizeof(b_data_type) * runtime_ldb * ((b_arrangement == arrangement::col_major) ? k_size : const_max(m_size, n_size)));
                const unsigned int size_ipiv = smem_align(sizeof(int) * const_min(m_size, n_size));
                const unsigned int size_tau  = smem_align(sizeof(a_data_type) * const_min(m_size, n_size));
                const unsigned int size_work = smem_align(sizeof(a_data_type) * get_workspace_size());

                switch (base_type::this_solver_function_v) {
                    case function::potrf:
                    case function::getrf_no_pivot:
                        return batches_per_block * size_a;

                    case function::potrs:
                    case function::posv:
                    case function::getrs_no_pivot:
                    case function::gesv_no_pivot:
                        return batches_per_block * (size_a + size_b);

                    case function::getrf_partial_pivot:
                        return batches_per_block * (size_a + size_ipiv);

                    case function::getrs_partial_pivot:
                    case function::gesv_partial_pivot:
                        return batches_per_block * (size_a + size_b + size_ipiv);

                    case function::gtsv_no_pivot: {
                        const unsigned int size_a_act = 2 * smem_align(sizeof(a_data_type) * (m_size - 1)) + smem_align(sizeof(a_data_type) * m_size);
                        return batches_per_block * (size_a_act + size_b);
                    }

                    case function::geqrf:
                    case function::gelqf:
                    case function::modified_lu:
                        return batches_per_block * (size_a + size_tau);

                    case function::gels:
                        return batches_per_block * (size_a + size_tau + size_b);

                    // trsm uses m, n, k differently from most routines.
                    // B (m, n). A (m, m) if left and (n, n) if right. k_size is ignored
                    case function::trsm: {
                        const auto AM         = (side == side::left) ? m_size : n_size;
                        const auto size_a_act = smem_align(sizeof(a_data_type) * runtime_lda * AM);
                        const auto size_b_act = smem_align(sizeof(b_data_type) * runtime_ldb * ((b_arrangement == arrangement::col_major) ? n_size : m_size));

                        return batches_per_block * (size_a_act + size_b_act);
                    }

                    // unmqr and unmlq use m, n, k differently from most routines.
                    // Matrix C (m, n)
                    // For unmqr: Matrix A/Q (m, k) if left and (n, k) if right
                    // For unmlq: Matrix A/Q (k, m) if left and (k, n) if right
                    case function::unmqr:
                    case function::unmlq: {
                        const bool         is_left      = (side == cusolverdx::side::left);
                        const bool         is_qr        = base_type::this_solver_function_v == function::unmqr;
                        const auto         AM           = is_qr ? (is_left ? m_size : n_size) : k_size;
                        const auto         AN           = is_qr ? k_size : (is_left ? m_size : n_size);
                        const auto         BM           = m_size;
                        const auto         BN           = n_size;
                        const unsigned int size_a_act   = smem_align(sizeof(a_data_type) * runtime_lda * ((a_arrangement == arrangement::col_major) ? AN : AM));
                        const unsigned int size_b_act   = smem_align(sizeof(b_data_type) * runtime_ldb * ((b_arrangement == arrangement::col_major) ? BN : BM));
                        const unsigned int size_tau_act = smem_align(sizeof(a_data_type) * k_size);

                        return batches_per_block * (size_a_act + size_tau_act + size_b_act);
                    }

                    case function::ungqr:
                    case function::unglq: {
                        const unsigned int size_tau_act = smem_align(sizeof(a_data_type) * k_size);
                        return batches_per_block * (size_a + size_tau_act);
                    }

                    // htev and bdsvd use m, n, k differently from most routines.
                    // Matrix A is either bidiagonal or symmetric tridiagonal.
                    // D array is of size m, and E array is of size m-1
                    // n and k are ignored
                    case function::htev:
                        // Need n words for the diagonal and n-1 words for the off-diagonal
                        // If vectors are requested, need n*n additional words
                        return batches_per_block * (smem_align(sizeof(a_data_type) * m_size) + smem_align(sizeof(a_data_type) * (m_size - 1)) +
                                                    (job == cusolverdx::job::no_vectors ? 0 : smem_align(sizeof(a_data_type) * m_size * runtime_lda)));


                    case function::heev: {
                        const unsigned int size_evals = smem_align(sizeof(a_precision) * m_size);
                        // size_work already includes BPB
                        return batches_per_block * (size_a + size_evals) + size_work;
                    }

                    case function::hegst: {
                        const unsigned int hegst_size_b = smem_align(sizeof(b_data_type) * runtime_ldb * m_size);
                        return batches_per_block * (size_a + hegst_size_b) + size_work;
                    }

                    case function::hegv: {
                        const unsigned int hegv_size_b = smem_align(sizeof(b_data_type) * runtime_ldb * m_size);
                        const unsigned int size_evals  = smem_align(sizeof(a_precision) * m_size);
                        return batches_per_block * (size_a + hegv_size_b + size_evals) + size_work;
                    }

                    case function::bdsvd: {
                        const unsigned int size_u     = (jobu != job::no_vectors) ? smem_align(sizeof(a_data_type) * runtime_lda * m_size) : 0;
                        const unsigned int size_vt    = (jobvt != job::no_vectors) ? smem_align(sizeof(a_data_type) * runtime_ldb * m_size) : 0;
                        // Need n words for the diagonal and n-1 words for the off-diagonal
                        return batches_per_block * (smem_align(sizeof(a_data_type) * m_size) + smem_align(sizeof(a_data_type) * (m_size - 1)) +
                                                    size_u + size_vt);
                    }

                    case function::gesvd: {
                        const unsigned int size_svals = smem_align(sizeof(a_precision) * const_min(m_size, n_size));

                        const unsigned int m_u    = m_size;
                        const unsigned int n_u    = jobu == job::all_vectors ? m_size : const_min(m_size, n_size);
                        const unsigned int m_vt   = jobvt == job::all_vectors ? n_size : const_min(m_size, n_size);
                        const unsigned int n_vt   = n_size;

                        const unsigned int size_u = (jobu == job::no_vectors || jobu == job::overwrite_vectors) ? 0
                                                    : smem_align(sizeof(a_data_type) * runtime_ldb * ((b_arrangement == arrangement::col_major) ? n_u : m_u));
                        const unsigned int size_vt = (jobvt == job::no_vectors || jobvt == job::overwrite_vectors) ? 0
                                                    : smem_align(sizeof(a_data_type) * runtime_ldc * ((c_arrangement == arrangement::col_major) ? n_vt : m_vt));
                        // size_work already includes BPB
                        return batches_per_block * (size_a + size_svals + size_u + size_vt) + size_work;
                    }

                    default:
                        // Unknown routine
                        return 0;
                }
            }

            // trsm
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::trsm, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                static_assert(base_type::has_side, "trsm requires a side");
                static_assert(base_type::has_diag, "trsm requires a diagonal mode");
                static_assert(base_type::has_fill_mode, "trsm requires a fill mode");

                trsm::block_execute<a_cuda_data_type, m_size, n_size, side, diag, transpose, fill_mode, a_arrangement, b_arrangement, max_threads_per_block, batches_per_block>(
                    (const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // potrf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrf, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, status_type* status) {
                static_assert(base_type::has_fill_mode, "potrf requires a fill mode");

                potrf::block_execute<a_cuda_data_type, m_size, fill_mode, a_arrangement, max_threads_per_block, batches_per_block, sm>((a_cuda_data_type*)A, runtime_lda, status, get_thread_id());
            }

            // potrs
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrs, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                static_assert(base_type::has_fill_mode, "potrs requires a fill mode");

                potrs::block_execute<a_cuda_data_type, m_size, k_size, fill_mode, a_arrangement, b_arrangement, max_threads_per_block, batches_per_block>(
                    (const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // posv
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb, status_type* status) {
                static_assert(base_type::has_fill_mode, "posv requires a fill mode");

                posv::block_execute<a_cuda_data_type, m_size, k_size, fill_mode, a_arrangement, b_arrangement, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, status, get_thread_id());
            }

            // getrf_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_no_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, status_type* status) {

                getrf_no_pivot::block_execute<a_cuda_data_type, m_size, n_size, a_arrangement, max_threads_per_block, batches_per_block, sm>((a_cuda_data_type*)A, runtime_lda, status, get_thread_id());
            }

            // modified_lu
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::modified_lu, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* S) {
                getrf_no_pivot::block_execute<a_cuda_data_type, m_size, n_size, a_arrangement, max_threads_per_block, batches_per_block, sm, true /**is_modified_lu*/>(
                    (a_cuda_data_type*)A, runtime_lda, nullptr, get_thread_id(), (a_cuda_data_type*)S);
            }

            // getrs_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrs_no_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                static_assert(m_size == n_size, "getrs requires M=N");

                getrs_no_pivot::block_execute<a_cuda_data_type, m_size, k_size, a_arrangement, b_arrangement, transpose, max_threads_per_block, batches_per_block>(
                    (const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // gesv_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_no_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb, status_type* status) {
                static_assert(m_size == n_size, "gesv requires M=N");

                gesv_no_pivot::block_execute<a_cuda_data_type, m_size, k_size, a_arrangement, b_arrangement, transpose, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, status, get_thread_id());
            }

            // getrf_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_partial_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, int* ipiv, status_type* status) {

                getrf_partial_pivot::block_execute<a_cuda_data_type, m_size, n_size, a_arrangement, max_threads_per_block, batches_per_block, sm>((a_cuda_data_type*)A, runtime_lda, ipiv, status, get_thread_id());
            }

            // getrs_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrs_partial_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, const int* ipiv, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                static_assert(m_size == n_size, "getrs requires M=N");

                getrs_partial_pivot::block_execute<a_cuda_data_type, m_size, k_size, a_arrangement, b_arrangement, transpose, max_threads_per_block, batches_per_block>(
                    (const a_cuda_data_type*)A, runtime_lda, ipiv, (b_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // gesv_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, int* ipiv, b_data_type* B, const unsigned int runtime_ldb, status_type* status) {
                static_assert(m_size == n_size, "gesv requires M=N");

                gesv_partial_pivot::block_execute<a_cuda_data_type, m_size, k_size, a_arrangement, b_arrangement, transpose, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)A, runtime_lda, ipiv, (a_cuda_data_type*)B, runtime_ldb, status, get_thread_id());
            }

            // gtsv_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gtsv_no_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* dl, const a_data_type* d, const a_data_type* du, a_data_type* b, const unsigned int runtime_ldb, status_type* status) {
                static_assert(m_size == n_size, "gtsv requires M=N");

                gtsv_no_pivot::block_execute<a_cuda_data_type, m_size, k_size, b_arrangement, max_threads_per_block, batches_per_block, sm>(
                    (const a_cuda_data_type*)dl, (const a_cuda_data_type*)d, (const a_cuda_data_type*)du, (a_cuda_data_type*)b, runtime_ldb, status, get_thread_id());
            }

            // geqrf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::geqrf, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau) {

                geqrf::block_execute<a_cuda_data_type, m_size, n_size, a_arrangement, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, get_thread_id());
            }

            // gelqf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gelqf, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau) {

                gelqf::block_execute<a_cuda_data_type, m_size, n_size, a_arrangement, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, get_thread_id());
            }

            // unmqr
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmqr, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb) {
                static_assert(base_type::has_side, "unmqr requires a side");

                unmqr::block_execute<a_cuda_data_type, m_size, n_size, k_size, side, transpose, a_arrangement, b_arrangement, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // unmlq
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmlq, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb) {
                static_assert(base_type::has_side, "unmlq requires a side");

                unmlq::block_execute<a_cuda_data_type, m_size, n_size, k_size, side, transpose, a_arrangement, b_arrangement, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // ungqr
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::ungqr, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau) {
                ungqr::block_execute<a_cuda_data_type, m_size, n_size, k_size, a_arrangement, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, get_thread_id());
            }

            // unglq
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unglq, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau) {
                unglq::block_execute<a_cuda_data_type, m_size, n_size, k_size, a_arrangement, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, get_thread_id());
            }

            // gels
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gels, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb) {
                gels::block_execute<a_cuda_data_type, m_size, n_size, k_size, a_arrangement, b_arrangement, transpose, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // htev - optional vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::htev, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, a_data_type* v, int ldv, status_type* status) {
                htev::block_execute<a_cuda_data_type, a_cuda_data_type, m_size, job, a_arrangement, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)d, (a_cuda_data_type*)e, (a_cuda_data_type*)v, ldv, status, get_thread_id());
            }

            // htev - no vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::htev, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, status_type* status) {
                static_assert(job == job::no_vectors, "To compute vectors with htev, a vector array must be provided.");

                execute(d, e, nullptr, m_size, status);
            }

            // heev
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::heev, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_precision* lambda, a_data_type* workspace, status_type* status) {
                static_assert(base_type::has_fill_mode, "heev requires a fill mode");

                heev::block_execute<a_cuda_data_type, m_size, fill_mode, a_arrangement, job, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)A, runtime_lda, lambda, (a_cuda_data_type*)workspace, status, get_thread_id());
            }

            // hegst
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegst, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, const a_data_type* B, const unsigned int runtime_ldb, a_data_type* workspace) {
                static_assert(base_type::has_fill_mode, "hegst requires a fill mode");

                hegst::block_execute<a_cuda_data_type, m_size, fill_mode, a_arrangement, b_arrangement, eig_type, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, (a_cuda_data_type*)workspace, get_thread_id());
            }

            // hegv
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegv, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda,
                                           a_data_type* B, const unsigned int runtime_ldb,
                                           a_precision* lambda,
                                           a_data_type* workspace,
                                           status_type* status) {
                static_assert(base_type::has_fill_mode, "hegv requires a fill mode");

                hegv::block_execute<a_cuda_data_type, m_size, fill_mode, a_arrangement, b_arrangement, eig_type, job, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, lambda, (a_cuda_data_type*)workspace, status, get_thread_id());
            }

            // bdsvd - vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::bdsvd, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, a_data_type* U, int ldu, a_data_type* VT, int ldvt, status_type* status) {
                bdsvd::block_execute<a_cuda_data_type, a_cuda_data_type, m_size, jobu, jobvt, a_arrangement, b_arrangement, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)d, (a_cuda_data_type*)e, (a_cuda_data_type*)U, ldu, (a_cuda_data_type*)VT, ldvt, status, get_thread_id());
            }

            // bdsvd - no vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::bdsvd, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, status_type* status) {
                static_assert(jobu == job::no_vectors && jobvt == job::no_vectors, "To compute vectors with bdsvd, vector arrays must be provided.");

                execute(d, e, nullptr, 0, nullptr, 0, status);
            }

            // gesvd - vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesvd, int> = 0>
            inline __device__ void execute(a_data_type*       A, const unsigned int runtime_lda,
                                           a_precision*       sigma,
                                           a_data_type*       U, const unsigned int runtime_ldu,
                                           a_data_type*       VT, const unsigned int runtime_ldvt,
                                           a_data_type*       workspace,
                                           status_type*       status) {
                gesvd::block_execute<a_cuda_data_type, m_size, n_size, jobu, jobvt, a_arrangement, b_arrangement, c_arrangement, max_threads_per_block, batches_per_block, sm>(
                    (a_cuda_data_type*)A, runtime_lda, sigma, (a_cuda_data_type*)U, runtime_ldu, (a_cuda_data_type*)VT, runtime_ldvt, (a_cuda_data_type*)workspace, status, get_thread_id());
            }

            // gesvd - no vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesvd, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_precision* sigma, a_data_type* workspace, status_type* status) {
                static_assert(jobu == job::no_vectors && jobvt == job::no_vectors, "To compute vectors with gesvd, vector arrays must be provided");

                execute(A, runtime_lda, sigma, nullptr, 0, nullptr, 0, workspace, status);
            }

            /* ********************************************************************** */
            /* *********************** End of functions ***************************** */
            /* ********************************************************************** */

            static constexpr unsigned int suggested_batches_per_block = get_suggested_batches_per_block();

            static constexpr dim3 suggested_block_dim = get_suggested_block_dim();
            static constexpr dim3 block_dim           = get_block_dim();

            static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
            static constexpr unsigned int min_blocks_per_multiprocessor = 1;

            static constexpr unsigned int workspace_size     = get_workspace_size();
            static constexpr unsigned int shared_memory_size = get_shared_memory_size();
        };
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_BLOCK_EXECUTION_HPP
