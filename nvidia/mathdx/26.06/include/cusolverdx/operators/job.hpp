// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_JOB_HPP
#define CUSOLVERDX_OPERATORS_JOB_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "cusolverdx/operators/operator_type.hpp"
#include "cusolverdx/operators/enums.hpp"

namespace cusolverdx {

    template<job ju, job jvt = ju>
    struct Job: commondx::detail::operator_expression {
        static_assert(ju == job::no_vectors || ju == job::all_vectors || ju == job::some_vectors || ju == job::multiply_vectors || ju == job::overwrite_vectors,
                      "Job U has to be no_vectors, all_vectors, some_vectors, multiply_vectors, or overwrite_vectors");
        static_assert(jvt == job::no_vectors || jvt == job::all_vectors || jvt == job::some_vectors || jvt == job::multiply_vectors ||
                          jvt == job::overwrite_vectors,
                      "Job VT has to be no_vectors, all_vectors, some_vectors, multiply_vectors, or overwrite_vectors");

        static constexpr job jobu = ju;
        static constexpr job jobvt = jvt;
    };

    namespace detail {
        using default_job_operator = Job<job::no_vectors, job::no_vectors>;
    } // namespace detail

} // namespace cusolverdx

namespace commondx::detail {
    template<cusolverdx::job ju, cusolverdx::job jvt>
    struct is_operator<cusolverdx::operator_type, cusolverdx::operator_type::job, cusolverdx::Job<ju, jvt>>: COMMONDX_STL_NAMESPACE::true_type {};

    template<cusolverdx::job ju, cusolverdx::job jvt>
    struct get_operator_type<cusolverdx::operator_type, cusolverdx::Job<ju, jvt>> {
        static constexpr cusolverdx::operator_type value = cusolverdx::operator_type::job;
    };
} // namespace commondx::detail

#endif // CUSOLVERDX_OPERATORS_JOB_HPP
