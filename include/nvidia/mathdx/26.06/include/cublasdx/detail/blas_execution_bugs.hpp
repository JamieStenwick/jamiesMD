#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_BUGS_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_BUGS_HPP

#include "cublasdx/detail/system_checks.hpp"

/// ---- Bug Checks

#if !defined(CUBLASDX_IGNORE_NVBUG_5668269_ASSERT) && AFFECTED_BY_NVBUG_5668269
    #define NVBUG_5668269_HANDLER \
            static constexpr bool can_be_impacted_by_nvbug_5668269 = \
                ((sm_v == 1000) || (sm_v == 1030) || (sm_v == 1100)) && \
                (cute::is_same_v<typename this_blas_precision::a_type, __half>) && \
                (cute::is_same_v<typename this_blas_precision::b_type, __half>) && \
                (this_blas_size::m == 181 and this_blas_size::n == 320 and this_blas_size::k == 36); \
                static_assert(not can_be_impacted_by_nvbug_5668269, \
                              "This configuration can be impacted by a CUDA 13.0 bug 5668269. \n" \
                              "Please use either CTK 13.1 or define CUBLASDX_IGNORE_NVBUG_5668269_ASSERT to manually " \
                              "verify results\n" \
                              "For more details please consult cuBLASDx documentation at " \
                              "https://docs.nvidia.com/cuda/cublasdx/index.html");
#else
#define NVBUG_5668269_HANDLER 
#endif

#if !defined(CUBLASDX_IGNORE_NVBUG_5430354_ASSERT) && AFFECTED_BY_NVBUG_5430354
    #define NVBUG_5430354_HANDLER \
            static constexpr bool can_be_impacted_by_nvbug_5430354_blackwell = \
                (sm_v >= 1000) && \
                (sizeof(typename this_blas_precision::a_type) == sizeof(int8_t) && \
                 sizeof(typename this_blas_precision::b_type) == sizeof(int8_t) && \
                 sizeof(typename this_blas_precision::c_type) == sizeof(int16_t)) && \
                (this_blas_size::m == 112 and this_blas_size::n == 136 and this_blas_size::k == 168) && has_ld; \
            static constexpr bool can_be_impacted_by_nvbug_5430354_blackwell_2 = \
                (sm_v == 1000 || sm_v == 1030 || sm_v == 1100) && \
                (sizeof(typename this_blas_precision::a_type) == sizeof(cublasdx::tfloat32_t) && \
                 sizeof(typename this_blas_precision::b_type) == sizeof(__half) && \
                 sizeof(typename this_blas_precision::c_type) == sizeof(double)) && \
                 base_type::this_blas_type_v == type::complex && \
                (this_blas_size::m == 105 and this_blas_size::n == 105 and this_blas_size::k == 105); \
            static constexpr bool can_be_impacted_by_nvbug_5430354 = \
                (sm_v == 800 or sm_v == 900) && commondx::is_floating_point_v<typename this_blas_precision::a_type> && \
                commondx::is_floating_point_v<typename this_blas_precision::b_type> && \
                commondx::is_floating_point_v<typename this_blas_precision::c_type> && \
                (sizeof(typename this_blas_precision::a_type) == sizeof(int8_t) || \
                 sizeof(typename this_blas_precision::b_type) == sizeof(int8_t)) && \
                sizeof(typename this_blas_precision::c_type) == sizeof(int16_t) && \
                base_type::this_blas_type_v == type::complex && \
                (this_blas_size::m % 16 != 0 || this_blas_size::n % 16 != 0 || this_blas_size::k % 16 != 0) && has_ld; \
            static_assert(not(can_be_impacted_by_nvbug_5430354 or can_be_impacted_by_nvbug_5430354_blackwell or can_be_impacted_by_nvbug_5430354_blackwell_2), \
                          "This configuration can be impacted by a CUDA 13.0+ bug 5430354. \n" \
                          "Please use either CTK 13.2 or define CUBLASDX_IGNORE_NVBUG_5430354_ASSERT to manually " \
                          "verify results\n" \
                          "For more details please consult cuBLASDx documentation at " \
                          "https://docs.nvidia.com/cuda/cublasdx/index.html");
#else
    #define NVBUG_5430354_HANDLER 
#endif

#if !defined(CUBLASDX_IGNORE_NVBUG_5697031_ASSERT) && AFFECTED_BY_NVBUG_5697031
    #define NVBUG_5697031_HANDLER \
            static constexpr bool can_be_impacted_by_nvbug_5697031 = \
                ((sm_v == 1000) || (sm_v == 1030) || (sm_v == 1100)) && \
                sizeof(typename this_blas_precision::a_type) == sizeof(int16_t) && \
                sizeof(typename this_blas_precision::b_type) == sizeof(int16_t) && \
                sizeof(typename this_blas_precision::c_type) == sizeof(int32_t) && \
                (this_blas_size::m == 158 and this_blas_size::n == 200 and this_blas_size::k == 144); \
            static_assert(not can_be_impacted_by_nvbug_5697031, \
                          "This configuration can be impacted by a CUDA 13.0 bug #5697031 \n" \
                          "Please use either an updated CTK or define CUBLASDX_IGNORE_NVBUG_5697031_ASSERT to manually " \
                          "verify results.");
#else
#define NVBUG_5697031_HANDLER 
#endif

#if !defined(CUBLASDX_IGNORE_NVBUG_6283425_ASSERT) && AFFECTED_BY_NVBUG_6283425
    #define NVBUG_6283425_HANDLER \
            static constexpr bool can_be_impacted_by_nvbug_6283425 = \
                cute::is_same_v<typename this_blas_precision::a_type, float> && \
                cute::is_same_v<typename this_blas_precision::b_type, float> && \
                cute::is_same_v<typename this_blas_precision::c_type, float> && \
                (this_blas_size::m == 167 and this_blas_size::n == 94 and this_blas_size::k == 36) && \
                (this_blas_block_dim_v.x == 32 and this_blas_block_dim_v.y == 1 and this_blas_block_dim_v.z == 1) && \
                (lda == 80 and ldb == 152 and ldc == 176); \
            static_assert(not can_be_impacted_by_nvbug_6283425, \
                          "This configuration can be impacted by a CUDA 13.1+ bug #6283425 \n" \
                          "Please use either an updated CTK or define CUBLASDX_IGNORE_NVBUG_6283425_ASSERT to manually " \
                          "verify results.");
#else
#define NVBUG_6283425_HANDLER
#endif

#if !defined(CUBLASDX_IGNORE_NVBUG_6283118_ASSERT) && AFFECTED_BY_NVBUG_6283118
    #define NVBUG_6283118_HANDLER \
            static constexpr bool can_be_impacted_by_nvbug_6283118 = \
                (sm_v == 900) && \
                cute::is_same_v<typename this_blas_precision::a_type, __nv_fp8_e4m3> && \
                cute::is_same_v<typename this_blas_precision::b_type, __nv_fp8_e5m2> && \
                cute::is_same_v<typename this_blas_precision::c_type, __half> && \
                (base_type::this_blas_type_v == type::complex) && \
                (base_type::this_blas_transpose_mode_a == transpose_mode::transposed) && \
                (base_type::this_blas_transpose_mode_b == transpose_mode::non_transposed) && \
                (this_blas_size::m == 124 and this_blas_size::n == 197 and this_blas_size::k == 67) && \
                (this_blas_block_dim_v.x == 128 and this_blas_block_dim_v.y == 1 and this_blas_block_dim_v.z == 1) && \
                (lda == 167 and ldb == 156 and ldc == 142); \
            static_assert(not can_be_impacted_by_nvbug_6283118, \
                          "This configuration can be impacted by a CUDA 13.1+ bug #6283118 \n" \
                          "Please use either an updated CTK or define CUBLASDX_IGNORE_NVBUG_6283118_ASSERT to manually " \
                          "verify results.");
#else
#define NVBUG_6283118_HANDLER
#endif

#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_BUGS_HPP
