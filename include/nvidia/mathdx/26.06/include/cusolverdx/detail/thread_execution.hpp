// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_THREAD_EXECUTION_HPP
#define CUSOLVERDX_DETAIL_THREAD_EXECUTION_HPP

#include "cusolverdx/detail/solver_execution.hpp"

namespace cusolverdx {
    namespace detail {

        template<class... Operators>
        class thread_execution: public solver_execution<Operators...> {
            using this_type = thread_execution<Operators...>;
            using base_type = solver_execution<Operators...>;

            // Import precision type from base class
            using typename base_type::this_solver_precision;

            /// ---- Constraints
            static_assert(base_type::has_thread, "Thread operator is missing in the solver description");

            static_assert(!has_operator_v<operator_type::block_dim, base_type>, "Solver for thread execution can't contain BlockDim<> operator");
            static_assert(!has_operator_v<operator_type::batches_per_block, base_type>, "Solver for thread execution can't contain BatchesPerBlock<> operator");
            static_assert(!has_operator_v<operator_type::blocks_per_cluster, base_type>, "Solver for thread execution can't contain BlocksPerCluster<> operator");
            static_assert(!has_operator_v<operator_type::tile_size, base_type>, "Solver for thread execution can't contain TileSize<> operator");

        public:
            static constexpr auto m_size = base_type::m_size;
            static constexpr auto n_size = base_type::n_size;
            static constexpr auto k_size = base_type::k_size;
            static constexpr auto lda    = base_type::lda;
            static constexpr auto ldb    = base_type::ldb;
            static constexpr auto ldc    = base_type::ldc;

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

            __host__ __device__ __forceinline__ static constexpr int get_workspace_size() {
                if constexpr (function_of_v<this_type> == function::heev) {
                    return heev::workspace_size_thread(dispatch_type, m_size, job);
                } else if constexpr (function_of_v<this_type> == function::hegst) {
                    return hegst::workspace_size_thread(dispatch_type, m_size, eig_type);
                } else if constexpr (function_of_v<this_type> == function::hegv) {
                    return hegv::workspace_size_thread(dispatch_type, m_size, eig_type, job);
                } else if constexpr (function_of_v<this_type> == function::gesvd) {
                    return gesvd::workspace_size_thread(dispatch_type, m_size, n_size, jobu, jobvt);
                } else {
                    return 0;
                }
            }

        public:
            // trsm
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::trsm, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                static_assert(base_type::has_side, "trsm requires a side");
                static_assert(base_type::has_diag, "trsm requires a diagonal mode");
                static_assert(base_type::has_fill_mode, "trsm requires a fill mode");

                trsm::thread_execute<a_cuda_data_type, m_size, n_size, side, diag, transpose, fill_mode, a_arrangement, b_arrangement>(
                    (const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb);
            }

            // potrf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrf, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, status_type* status) {
                static_assert(base_type::has_fill_mode, "potrf requires a fill mode");

                potrf::thread_execute<a_cuda_data_type, m_size, fill_mode, a_arrangement>((a_cuda_data_type*)A, runtime_lda, status);
            }

            // potrs
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrs, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                static_assert(base_type::has_fill_mode, "potrs requires a fill mode");

                potrs::thread_execute<a_cuda_data_type, m_size, k_size, fill_mode, a_arrangement, b_arrangement>((const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb);
            }

            // posv
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb, status_type* status) {
                static_assert(base_type::has_fill_mode, "posv requires a fill mode");

                posv::thread_execute<a_cuda_data_type, m_size, k_size, fill_mode, a_arrangement, b_arrangement>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, status);
            }

            // getrf_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_no_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, status_type* status) {

                getrf_no_pivot::thread_execute<a_cuda_data_type, m_size, n_size, a_arrangement>((a_cuda_data_type*)A, runtime_lda, status);
            }

            // modified_lu
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::modified_lu, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* S) {
                getrf_no_pivot::thread_execute<a_cuda_data_type, m_size, n_size, a_arrangement, true /**is_modified_lu*/>((a_cuda_data_type*)A, runtime_lda, nullptr, (a_cuda_data_type*)S);
            }

            // getrs_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrs_no_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                static_assert(m_size == n_size, "getrs requires M=N");

                getrs_no_pivot::thread_execute<a_cuda_data_type, m_size, k_size, a_arrangement, b_arrangement, transpose>((const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb);
            }
            
            // gesv_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_no_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb, status_type* status) {
                static_assert(m_size == n_size, "gesv requires M=N");

                gesv_no_pivot::thread_execute<a_cuda_data_type, m_size, k_size, a_arrangement, b_arrangement, transpose>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, status);
            }

            // getrf_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_partial_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, int* ipiv, status_type* status) {

                getrf_partial_pivot::thread_execute<a_cuda_data_type, m_size, n_size, a_arrangement>((a_cuda_data_type*)A, runtime_lda, ipiv, status);
            }

            // getrs_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrs_partial_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, const int* ipiv, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                static_assert(m_size == n_size, "getrs requires M=N");

                getrs_partial_pivot::thread_execute<a_cuda_data_type, m_size, k_size, a_arrangement, b_arrangement, transpose>(
                    (const a_cuda_data_type*)A, runtime_lda, ipiv, (b_cuda_data_type*)B, runtime_ldb);
            }

            // gesv_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, int* ipiv, b_data_type* B, const unsigned int runtime_ldb, status_type* status) {
                static_assert(m_size == n_size, "gesv requires M=N");

                gesv_partial_pivot::thread_execute<a_cuda_data_type, m_size, k_size, a_arrangement, b_arrangement, transpose>(
                    (a_cuda_data_type*)A, runtime_lda, ipiv, (a_cuda_data_type*)B, runtime_ldb, status);
            }

            //gtsv_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gtsv_no_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* dl, const a_data_type* d, const a_data_type* du, a_data_type* b, const unsigned int runtime_ldb, status_type* status) {
                static_assert(m_size == n_size, "gtsv requires M=N");

                gtsv_no_pivot::thread_execute<a_cuda_data_type, m_size, k_size, b_arrangement>(
                    (const a_cuda_data_type*)dl, (const a_cuda_data_type*)d, (const a_cuda_data_type*)du, (a_cuda_data_type*)b, runtime_ldb, status);
            }

            // geqrf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::geqrf, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau) {

                geqrf::thread_execute<a_cuda_data_type, m_size, n_size, a_arrangement>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau);
            }

            // gelqf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gelqf, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau) {

                gelqf::thread_execute<a_cuda_data_type, m_size, n_size, a_arrangement>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau);
            }

            // unmqr
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmqr, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb) {
                static_assert(base_type::has_side, "unmqr requires a side");

                unmqr::thread_execute<a_cuda_data_type, m_size, n_size, k_size, side, transpose, a_arrangement, b_arrangement>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, (a_cuda_data_type*)B, runtime_ldb);
            }

            // unmlq
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmlq, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb) {
                static_assert(base_type::has_side, "unmlq requires a side");

                unmlq::thread_execute<a_cuda_data_type, m_size, n_size, k_size, side, transpose, a_arrangement, b_arrangement>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, (a_cuda_data_type*)B, runtime_ldb);
            }

            // ungqr
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::ungqr, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau) {
                ungqr::thread_execute<a_cuda_data_type, m_size, n_size, k_size, a_arrangement>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau);
            }

            // unglq
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unglq, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau) {
                unglq::thread_execute<a_cuda_data_type, m_size, n_size, k_size, a_arrangement>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau);
            }

            // gels
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gels, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb) {
                gels::thread_execute<a_cuda_data_type, m_size, n_size, k_size, a_arrangement, b_arrangement, transpose>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, (a_cuda_data_type*)B, runtime_ldb);
            }

            // htev - optional vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::htev, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, a_data_type* v, int ldv, status_type* status) {

                htev::thread_execute<a_cuda_data_type, a_cuda_data_type, m_size, job, a_arrangement>((a_cuda_data_type*)d, (a_cuda_data_type*)e, (a_cuda_data_type*)v, ldv, status);
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

                heev::thread_execute<a_cuda_data_type, m_size, fill_mode, a_arrangement, job>((a_cuda_data_type*)A, runtime_lda, lambda, (a_cuda_data_type*)workspace, status);
            }

            // hegst
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegst, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, const a_data_type* B, const unsigned int runtime_ldb, a_data_type* workspace) {
                static_assert(base_type::has_fill_mode, "hegst requires a fill mode");

                hegst::thread_execute<a_cuda_data_type, m_size, fill_mode, a_arrangement, b_arrangement, eig_type>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, (a_cuda_data_type*)workspace);
            }

            // hegv
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegv, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda,
                                           a_data_type* B, const unsigned int runtime_ldb,
                                           a_precision* lambda,
                                           a_data_type* workspace,
                                           status_type* status) {
                static_assert(base_type::has_fill_mode, "hegv requires a fill mode");

                hegv::thread_execute<a_cuda_data_type, m_size, fill_mode, a_arrangement, b_arrangement, eig_type, job>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, lambda, (a_cuda_data_type*)workspace, status);
            }

            // bdsvd - vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::bdsvd, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, a_data_type* U, int ldu, a_data_type* VT, int ldvt, status_type* status) {

                bdsvd::thread_execute<a_cuda_data_type, a_cuda_data_type, m_size, jobu, jobvt, a_arrangement, b_arrangement>((a_cuda_data_type*)d, (a_cuda_data_type*)e, (a_cuda_data_type*)U, ldu, (a_cuda_data_type*)VT, ldvt, status);
            }

            // bdsvd - no vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::bdsvd, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, status_type* status) {
                static_assert(jobu == job::no_vectors && jobvt == job::no_vectors, "To compute vectors with bdsvd, vector arrays must be provided.");

                execute(d, e, nullptr, 0, nullptr, 0, status);
            }

            // gesvd - vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesvd, int> = 0>
            inline __device__ void execute(a_data_type*       A,
                                           const unsigned int runtime_lda,
                                           a_precision*       sigma,
                                           a_data_type*       U,
                                           const unsigned int runtime_ldu,
                                           a_data_type*       VT,
                                           const unsigned int runtime_ldvt,
                                           a_data_type*       workspace,
                                           status_type*       status) {

                gesvd::thread_execute<a_cuda_data_type, m_size, n_size, jobu, jobvt, a_arrangement, b_arrangement, c_arrangement>(
                    (a_cuda_data_type*)A, runtime_lda, sigma, (a_cuda_data_type*)U, runtime_ldu, (a_cuda_data_type*)VT, runtime_ldvt, (a_cuda_data_type*)workspace, status);
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

            static constexpr unsigned int workspace_size = get_workspace_size();
        };
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_THREAD_EXECUTION_HPP
