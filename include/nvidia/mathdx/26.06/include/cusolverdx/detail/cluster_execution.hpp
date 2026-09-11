// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_CLUSTER_EXECUTION_HPP
#define CUSOLVERDX_DETAIL_CLUSTER_EXECUTION_HPP

#include "cusolverdx/detail/solver_execution.hpp"
#include "cusolverdx/database/potrf_cluster.cuh"

namespace cusolverdx {
    namespace detail {

        template<class... Operators>
        class cluster_execution: public solver_execution<Operators...> {
            using this_type = cluster_execution<Operators...>;
            using base_type = solver_execution<Operators...>;

            // Cluster execution only supports potrf for now
            static_assert(base_type::this_solver_function_v == function::potrf, "Cluster execution only supports potrf currently");

            // Import precision type from base class
            using typename base_type::this_solver_precision;

            /// ---- Constraints
            static_assert(base_type::has_cluster, "Cluster operator is missing in the solver description");
            static_assert(!has_operator_v<operator_type::batches_per_block, base_type>, "Solver for cluster execution can't contain BatchesPerBlock<> operator");
            static_assert(!base_type::has_sm || base_type::this_sm_v >= 900, "Cluster execution only supports Hopper (SM 90) or later");

        public:
            static constexpr auto m_size = base_type::m_size;
            static constexpr auto n_size = base_type::n_size;
            static constexpr auto k_size = base_type::k_size;
            static constexpr auto lda    = base_type::lda;
            static constexpr auto ldb    = base_type::ldb;
            static constexpr auto ldc    = base_type::ldc;

            static constexpr auto blocks_per_cluster = base_type::this_solver_blocks_per_cluster_v;

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
            
            __host__ __device__ __forceinline__ static constexpr dim3 get_suggested_block_dim() {
                static_assert(base_type::is_complete_v, "Can't provide suggested block dimensions, description is not complete");
                if constexpr (base_type::is_function_cholesky) {
                    constexpr auto tile_size = get_tile_size();
                    return potrf::suggested_cluster_block_dim(dispatch_type, m_size, tile_size, blocks_per_cluster, sm);
                } else {
                    return 128;
                }
            }

            __host__ __device__ __forceinline__ static constexpr dim3 get_block_dim() {
                static_assert(base_type::is_complete_v, "Can't provide block dimensions, description is not complete");
                if constexpr (base_type::has_block_dim) {
                    return base_type::this_block_dim_v;
                }
                return get_suggested_block_dim();
            }

            __device__ __forceinline__ unsigned get_thread_id() {
                const auto dim = get_block_dim();
                __builtin_assume(threadIdx.x < dim.x);
                __builtin_assume(threadIdx.y < dim.y);
                __builtin_assume(threadIdx.z < dim.z);

                return threadIdx.x + dim.x * (threadIdx.y + dim.y * threadIdx.z);
            }

        public:
            // Cluster execution expects A to point at the triangular matrix tiles in distributed shared memory.
            inline static constexpr unsigned int get_shared_memory_size() {
                static_assert(base_type::is_complete_v, "Can't calculate shared memory, description is not complete");

                switch (base_type::this_solver_function_v) {
                    case function::potrf: {
                        return potrf::shared_memory_size_cluster<a_cuda_data_type, m_size, get_tile_size(), blocks_per_cluster, sm>();
                    }
                    default:
                        return 0;
                }
            }
            inline static constexpr unsigned int get_suggested_tile_size() {
                static_assert(base_type::is_complete_v, "Can't provide suggested tile size, description is not complete");

                switch (base_type::this_solver_function_v) {
                    case function::potrf: {
                        return potrf::suggested_cluster_tile_size(dispatch_type, m_size, blocks_per_cluster, sm);
                    }
                    default:
                        return 64;
                }
            }
            inline static constexpr unsigned int get_tile_size() {
                if constexpr (base_type::has_tile_size) {
                    return base_type::this_solver_tile_size_v;
                }
                return get_suggested_tile_size();
            }

            // potrf
            template<class Solver = this_type, COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrf, int> = 0>
            inline __device__ void execute(a_data_type* A, const unsigned int runtime_lda, status_type* status) {
                static_assert(base_type::has_fill_mode, "potrf requires a fill mode");

                // Avoid a GEMM bug in cuBLASDx if block dim = 64 when using remote shared memory for certain datatype
                static_assert(!(type == cusolverdx::type::complex &&
                    same_type_v<a_precision, double> && max_threads_per_block < 128), "Cluster POTRF: complex double data type requires BlockDim<> of at least 128 threads.");

                (void)runtime_lda; // A is in cluster tiles in shared memory, so runtime_lda is not used
                potrf::cluster_execute<a_cuda_data_type,
                                       m_size,
                                       fill_mode,
                                       a_arrangement,
                                       max_threads_per_block,
                                       tile_size,
                                       blocks_per_cluster,
                                       sm>((a_cuda_data_type*)A, status, get_thread_id());
            }

            /* ********************************************************************** */
            /* *********************** End of functions ***************************** */
            /* ********************************************************************** */

            static constexpr dim3 suggested_block_dim = get_suggested_block_dim();
            static constexpr dim3 block_dim           = get_block_dim();

            static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
            static constexpr unsigned int min_blocks_per_multiprocessor = 1;

            static constexpr unsigned int suggested_tile_size = get_suggested_tile_size();
            static constexpr unsigned int tile_size           = get_tile_size();
            static constexpr unsigned int shared_memory_size  = get_shared_memory_size();
        };
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_CLUSTER_EXECUTION_HPP
