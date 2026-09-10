// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_SOLVER_EXECUTION_HPP
#define CUSOLVERDX_DETAIL_SOLVER_EXECUTION_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "cusolverdx/detail/solver_description.hpp"
#include "cusolverdx/detail/util.hpp"
#include "cusolverdx/detail/system_checks.hpp"

#include "cusolverdx/database/all.hpp"

namespace cusolverdx {
    namespace detail {

        // Forward declarations
        template<class... Operators>
        class thread_execution;
        template<class... Operators>
        class block_execution;
        template<class... Operators>
        class cluster_execution;

        inline static constexpr unsigned smem_align(const unsigned size, const unsigned alignment = 16) { return (size + alignment - 1) / alignment * alignment; }

        template<class... Operators>
        class solver_execution: public solver_description<Operators...>, public commondx::detail::execution_description_expression {
            using base_type = solver_description<Operators...>;
            using this_type = solver_execution<Operators...>;

        protected:
            // Precision type
            using typename base_type::this_solver_precision;

            // Value type
            using this_solver_data_type = map_value_type<base_type::this_solver_type_v, this_solver_precision>;

        public:
            static constexpr bool is_thread_execution  = has_operator_v<operator_type::thread, this_type>;
            static constexpr bool is_block_execution   = has_operator_v<operator_type::block, this_type>;
            static constexpr bool is_cluster_execution = has_operator_v<operator_type::cluster, this_type>;
            static_assert(static_cast<int>(is_thread_execution) + static_cast<int>(is_block_execution) + static_cast<int>(is_cluster_execution) == 1,
                          "A SOLVER description must have exactly one execution operator (Thread, Block, or Cluster)");

            using a_data_type = typename this_solver_data_type::a_type;
            using x_data_type = typename this_solver_data_type::x_type;
            using b_data_type = typename this_solver_data_type::b_type;

            using a_cuda_data_type = typename convert_to_cuda_type<a_data_type>::type;
            using b_cuda_data_type = typename convert_to_cuda_type<b_data_type>::type;
            using x_cuda_data_type = typename convert_to_cuda_type<x_data_type>::type;

            using a_precision = typename base_type::this_solver_precision::a_type;
            using b_precision = typename base_type::this_solver_precision::b_type;
            using x_precision = typename base_type::this_solver_precision::x_type;

            using status_type = int;

            static constexpr auto m_size = base_type::this_solver_size::m;
            static constexpr auto n_size = base_type::this_solver_size::n;
            static constexpr auto k_size = base_type::this_solver_size::k;
            static constexpr auto lda    = base_type::this_solver_lda;
            static constexpr auto ldb    = base_type::this_solver_ldb;
            static constexpr auto ldc    = base_type::this_solver_ldc;

            static constexpr auto a_size = base_type::this_solver_a_size;
            static constexpr auto b_size = base_type::this_solver_b_size;

            static constexpr bool is_function_cholesky         = base_type::is_function_cholesky;
            static constexpr bool is_function_lu               = base_type::is_function_lu;
            static constexpr bool is_function_lu_no_pivot      = base_type::is_function_lu_no_pivot;
            static constexpr bool is_function_lu_partial_pivot = base_type::is_function_lu_partial_pivot;
            static constexpr bool is_function_qr               = base_type::is_function_qr;
            static constexpr bool is_function_unmq             = base_type::is_function_unmq;
            static constexpr bool is_function_ungq             = base_type::is_function_ungq;
            static constexpr bool is_function_symmetric_eigen  = base_type::is_function_symmetric_eigen;
            static constexpr bool has_both_a_and_b_matrices    = base_type::has_both_a_and_b_matrices;
            static constexpr bool is_function_trsm             = base_type::is_function_trsm;
            static constexpr bool is_function_gtsv_no_pivot    = base_type::is_function_gtsv_no_pivot;

            //======================= Convenience overload helpers for adding lda/ldb from operator
            // potrf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrf, int> = 0>
            inline __device__ void execute(a_data_type* A, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, status);
                } else if constexpr (is_cluster_execution) {
                    static_cast<cluster_execution<Operators...>*>(this)->execute(A, lda, status);
                }
            }
            // getrf_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_no_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, status);
                }
            }
            // modified_lu
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::modified_lu, int> = 0>
            inline __device__ void execute(a_data_type* A, a_data_type* S) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, S);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, S);
                }
            }

            // getrf_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_partial_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, int* ipiv, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, ipiv, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, ipiv, status);
                }
            }

            // trsm, potrs, getrs_no_pivot
            template<
                class Solver = this_type,
                COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::trsm || function_of_v<Solver> == function::potrs || function_of_v<Solver> == function::getrs_no_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* A, b_data_type* B, unsigned int runtime_ldb = ldb) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, B, runtime_ldb);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, B, runtime_ldb);
                }
            }
            // getrs_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrs_partial_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* A, const int* ipiv, b_data_type* B, unsigned int runtime_ldb = ldb) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, ipiv, B, runtime_ldb);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, ipiv, B, runtime_ldb);
                }
            }

            // posv, gesv_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv || function_of_v<Solver> == function::gesv_no_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, b_data_type* B, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, B, ldb, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, B, ldb, status);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv || function_of_v<Solver> == function::gesv_no_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, b_data_type* B, const unsigned int runtime_ldb, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, B, runtime_ldb, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, B, runtime_ldb, status);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv || function_of_v<Solver> == function::gesv_no_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, b_data_type* B, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, runtime_lda, B, ldb, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, runtime_lda, B, ldb, status);
                }
            }
            // gesv_partial_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, int* ipiv, b_data_type* B, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, ipiv, B, ldb, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, ipiv, B, ldb, status);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, int* ipiv, b_data_type* B, unsigned int runtime_ldb, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, ipiv, B, runtime_ldb, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, ipiv, B, runtime_ldb, status);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, int* ipiv, b_data_type* B, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, runtime_lda, ipiv, B, ldb, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, runtime_lda, ipiv, B, ldb, status);
                }
            }

            // gtsv_no_pivot
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gtsv_no_pivot, int> = 0>
            inline __device__ void execute(const a_data_type* dl, const a_data_type* d, const a_data_type* du, a_data_type* b, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(dl, d, du, b, ldb, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(dl, d, du, b, ldb, status);
                }
            }

            // geqrf, gelqf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<(function_of_v<Solver> == function::geqrf || function_of_v<Solver> == function::gelqf), int> = 0>
            inline __device__ void execute(a_data_type* A, a_data_type* tau) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, tau);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, tau);
                }
            }

            // ungqr, unglq
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<(function_of_v<Solver> == function::ungqr || function_of_v<Solver> == function::unglq), int> = 0>
            inline __device__ void execute(a_data_type* A, const a_data_type* tau) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, tau);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, tau);
                }
            }

            // unmqr, unmlq
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmqr || function_of_v<Solver> == function::unmlq, int> = 0>
            inline __device__ void execute(const a_data_type* A, const a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, tau, B, runtime_ldb);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, tau, B, runtime_ldb);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmqr || function_of_v<Solver> == function::unmlq, int> = 0>
            inline __device__ void execute(const a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau, b_data_type* B) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, runtime_lda, tau, B, ldb);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, runtime_lda, tau, B, ldb);
                }
            }

            // gels
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gels, int> = 0>
            inline __device__ void execute(a_data_type* A, a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb = ldb) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, tau, B, runtime_ldb);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, tau, B, runtime_ldb);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gels, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau, b_data_type* B) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, runtime_lda, tau, B, ldb);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, runtime_lda, tau, B, ldb);
                }
            }

            // htev -- optional vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::htev, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, a_data_type* v, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(d, e, v, lda, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(d, e, v, lda, status);
                }
            }

            // heev
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::heev, int> = 0>
            inline __device__ void execute(a_data_type* A, a_precision* lambda, a_data_type* workspace, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, lambda, workspace, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, lambda, workspace, status);
                }
            }

            // hegst
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegst, int> = 0>
            inline __device__ void execute(a_data_type* A, const a_data_type* B, a_data_type* workspace) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, B, ldb, workspace);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, B, ldb, workspace);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegst, int> = 0>
            inline __device__ void execute(a_data_type* A, const a_data_type* B, const unsigned int runtime_ldb, a_data_type* workspace) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, B, runtime_ldb, workspace);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, B, runtime_ldb, workspace);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegst, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, const a_data_type* B, a_data_type* workspace) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, runtime_lda, B, ldb, workspace);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, runtime_lda, B, ldb, workspace);
                }
            }

            // hegv
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegv, int> = 0>
            inline __device__ void execute(a_data_type* A, a_data_type* B, a_precision* lambda, a_data_type* workspace, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, B, ldb, lambda, workspace, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, B, ldb, lambda, workspace, status);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegv, int> = 0>
            inline __device__ void execute(a_data_type* A, a_data_type* B, const unsigned int runtime_ldb, a_precision* lambda, a_data_type* workspace, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, B, runtime_ldb, lambda, workspace, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, B, runtime_ldb, lambda, workspace, status);
                }
            }
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::hegv, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* B, a_precision* lambda, a_data_type* workspace, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, runtime_lda, B, ldb, lambda, workspace, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, runtime_lda, B, ldb, lambda, workspace, status);
                }
            }

            // bdsvd -- optional vectors, API allowed are either all without runtime leading dimensions, or all with runtime leading dimensions.
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::bdsvd, int> = 0>
            inline __device__ void execute(a_data_type* d, a_data_type* e, a_data_type* U, a_data_type* VT, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(d, e, U, lda, VT, ldb, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(d, e, U, lda, VT, ldb, status);
                }
            }

            // gesvd with no vectors
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesvd, int> = 0>
            inline __device__ void execute(a_data_type* A, a_precision* sigma, a_data_type* workspace, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, sigma, workspace, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, sigma, workspace, status);
                }
            }
            // gesvd with vectors. API allowed are either all without runtime leading dimensions, or all with runtime leading dimensions.
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesvd, int> = 0>
            inline __device__ void execute(a_data_type* A, a_precision* sigma, a_data_type* U, a_data_type* V, a_data_type* workspace, status_type* status) {
                if constexpr (is_block_execution) {
                    static_cast<block_execution<Operators...>*>(this)->execute(A, lda, sigma, U, ldb, V, ldc, workspace, status);
                } else if constexpr (is_thread_execution) {
                    static_cast<thread_execution<Operators...>*>(this)->execute(A, lda, sigma, U, ldb, V, ldc, workspace, status);
                }
            }
        };

    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_SOLVER_EXECUTION_HPP
