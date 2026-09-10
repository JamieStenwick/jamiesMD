// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_BLOCK_EXECUTION_BUGS_HPP
#define CUSOLVERDX_DETAIL_BLOCK_EXECUTION_BUGS_HPP

/// ---- Bug checks for block execution (gesv_no_pivot, htev, bdsvd, gesvd)
/// When CUSOLVERDX_TEST_DISABLE_WARS is defined (CMake: CUSOLVERDX_TEST_DISABLE_WARS=ON for tests),
/// all static_assert handlers below are disabled so bug-exercising correctness tests can compile.


// CUDA 13.+ are impacted by NVBUG 5288270 (SM120 build)
#define AFFECTED_BY_NVBUG_5288270 (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ <= 3)

// HTEV/BDSVD wrong results: CTK 13.2+ on SM100+ (nvbug 5972531), CTK 13.0/13.1 on SM80/90 (nvbug 5986343)
#define AFFECTED_BY_NVBUG_5972531 (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
#define AFFECTED_BY_NVBUG_5986343 (__CUDACC_VER_MAJOR__ == 13 && (__CUDACC_VER_MINOR__ == 0 || __CUDACC_VER_MINOR__ == 1))

// GESVD wrong results on SM100+ with CTK 13.2+ (nvbug 5908003)
#define AFFECTED_BY_NVBUG_5908003 (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)


#if defined(CUSOLVERDX_TEST_DISABLE_WARS)

    #define CUSOLVERDX_NVBUG_5288270_GESV_NO_PIVOT_HANDLER(...)
    #define CUSOLVERDX_NVBUG_5972531_HTEV_HANDLER(...)
    #define CUSOLVERDX_NVBUG_5986343_BDSVD_HANDLER(...)
    #define CUSOLVERDX_NVBUG_5908003_GESVD_HANDLER(...)

#else

    #if !defined(CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT) && AFFECTED_BY_NVBUG_5288270
        #define CUSOLVERDX_NVBUG_5288270_GESV_NO_PIVOT_HANDLER(arch, is_real)                                             \
            static constexpr bool can_be_impacted_by_nvbug_5288270 = ((arch) == 1200 && (is_real));                      \
            static_assert(not can_be_impacted_by_nvbug_5288270,                                                           \
                          "This configuration can be impacted by CUDA 13.0, 13.1, 13.2, 13.3 on SM120 for bug 5288270. \n" \
                          "Please either update to the latest CUDA version, \n"                                           \
                          "or define CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT to ignore this check, \n"                    \
                          "add -Xptxas \"-O0\" to build your application, \n"                                            \
                          "and verify correctness of the results every time. \n"                                         \
                          "For more details see cuSolverDx release notes.");
    #else
        #define CUSOLVERDX_NVBUG_5288270_GESV_NO_PIVOT_HANDLER(arch, is_real)
    #endif

    #if !defined(CUSOLVERDX_IGNORE_NVBUG_5972531_ASSERT) && AFFECTED_BY_NVBUG_5972531
        #define CUSOLVERDX_NVBUG_5972531_HTEV_HANDLER(arch, is_real, job_val, n, nt)                                \
            static constexpr bool can_be_impacted_by_nvbug_5972531_htev =                                          \
                ((arch) >= 1000) && (is_real) && ((job_val) != job::no_vectors) &&                                  \
                ((n) == 7 || (n) == 15) && ((nt) == 64);                                                            \
            static_assert(not can_be_impacted_by_nvbug_5972531_htev,                                               \
                          "This HTEV configuration can be impacted by compiler bug 5972531 (SM100+ with CTK 13.2+). \n" \
                          "Please use CTK 13.0 or 13.1, or define CUSOLVERDX_IGNORE_NVBUG_5972531_ASSERT to manually verify results. \n" \
                          "For more details see cuSolverDx release notes.");
    #else
        #define CUSOLVERDX_NVBUG_5972531_HTEV_HANDLER(arch, is_real, job_val, n, nt)
    #endif

    #if !defined(CUSOLVERDX_IGNORE_NVBUG_5986343_ASSERT) && AFFECTED_BY_NVBUG_5986343
        #define CUSOLVERDX_NVBUG_5986343_BDSVD_HANDLER(arch, is_real, jobu_val, jobvt_val, n, nt)                       \
            static constexpr bool can_be_impacted_by_nvbug_5986343_bdsvd =                                              \
                ((arch) == 800 || (arch) == 860 || (arch) == 890 || (arch) == 900) &&                                   \
                (is_real) && ((jobu_val) != job::no_vectors && (jobvt_val) != job::no_vectors) &&                        \
                ((n) == 84) && ((nt) == 96);                                                                             \
            static_assert(not can_be_impacted_by_nvbug_5986343_bdsvd,                                                   \
                          "This BDSVD configuration can be impacted by compiler bug 5986343 (SM80/86/89/90 with CTK 13.0/13.1). \n" \
                          "Please use CTK 13.2+ or define CUSOLVERDX_IGNORE_NVBUG_5986343_ASSERT to manually verify results. \n" \
                          "For more details see cuSolverDx release notes.");
    #else
        #define CUSOLVERDX_NVBUG_5986343_BDSVD_HANDLER(arch, is_real, jobu_val, jobvt_val, n, nt)
    #endif

    #if !defined(CUSOLVERDX_IGNORE_NVBUG_5908003_ASSERT) && AFFECTED_BY_NVBUG_5908003
        #define CUSOLVERDX_NVBUG_5908003_GESVD_HANDLER(arch, is_real, jobu_val, jobvt_val, m, n)                         \
            static constexpr bool can_be_impacted_by_nvbug_5908003_gesvd =                                               \
                ((arch) >= 1000) && !(is_real) &&                                                                         \
                ((jobu_val) != job::no_vectors && (jobvt_val) != job::no_vectors) &&                                      \
                (((m) == 19 && (n) == 312) || ((m) == 10 && (n) == 16));                                                 \
            static_assert(not can_be_impacted_by_nvbug_5908003_gesvd,                                                    \
                          "This GESVD configuration can be impacted by nvbug 5908003 (SM100+ with CTK 13.2+). \n"        \
                          "Please use CTK 13.0 or 13.1, or define CUSOLVERDX_IGNORE_NVBUG_5908003_ASSERT to manually verify results. \n" \
                          "For more details see cuSolverDx release notes.");
    #else
        #define CUSOLVERDX_NVBUG_5908003_GESVD_HANDLER(arch, is_real, jobu_val, jobvt_val, m, n)
    #endif

#endif /* !CUSOLVERDX_TEST_DISABLE_WARS */

#endif // CUSOLVERDX_DETAIL_BLOCK_EXECUTION_BUGS_HPP
