// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_TRAITS_SOLVER_TRAITS_HPP
#define CUSOLVERDX_TRAITS_SOLVER_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/dx_traits.hpp"

#include "cusolverdx/detail/util.hpp"
#include "cusolverdx/detail/solver_description_fd.hpp"
#include "cusolverdx/operators.hpp"
#include "cusolverdx/types.hpp"
#include "cusolverdx/traits/detail/description_traits.hpp"
#include "cusolverdx/traits/detail/is_complete.hpp"

namespace cusolverdx {
    namespace detail {
        template<commondx::data_type data_type, class Precision>
        struct map_value_type {
            using a_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex), complex<typename Precision::a_type>, typename Precision::a_type>;
            using x_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex), complex<typename Precision::x_type>, typename Precision::x_type>;
            using b_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex), complex<typename Precision::b_type>, typename Precision::b_type>;
        };
    } // namespace detail

    // ------------------
    // Execution checkers
    // ------------------
    // is_thread
    template<class Description>
    struct is_thread {
    public:
        static constexpr bool value = detail::has_operator_v<operator_type::thread, Description>;
    };

    template<class Description>
    inline constexpr bool is_thread_v = is_thread<Description>::value;

    // is_block
    template<class Description>
    struct is_block {
    public:
        static constexpr bool value = detail::has_operator_v<operator_type::block, Description>;
    };

    template<class Description>
    inline constexpr bool is_block_v = is_block<Description>::value;

    // is_cluster
    template<class Description>
    struct is_cluster {
    public:
        static constexpr bool value = detail::has_operator_v<operator_type::cluster, Description>;
    };

    template<class Description>
    inline constexpr bool is_cluster_v = is_cluster<Description>::value;

    // ----------------
    // Operator getters
    // ----------------

    // precision_of
    template<class Description>
    struct precision_of {
    private:
        using description = commondx::detail::get_or_default_t<operator_type, operator_type::precision, Description, detail::default_precision_operator>;

    public:
        using a_type = typename description::a_type;
        using x_type = typename description::x_type;
        using b_type = typename description::b_type;
    };

    template<class Description>
    using precision_of_a_t = typename precision_of<Description>::a_type;
    template<class Description>
    using precision_of_x_t = typename precision_of<Description>::x_type;
    template<class Description>
    using precision_of_b_t = typename precision_of<Description>::b_type;

    // type_of
    template<class Description>
    using type_of = commondx::data_type_of<operator_type, Description, detail::default_type_operator>;
    template<class Description>
    inline constexpr type type_of_v = type_of<Description>::value;

    // sm_of
    template<class Description>
    using sm_of = commondx::sm_of<operator_type, Description>;
    template<class Description>
    inline constexpr unsigned int sm_of_v = sm_of<Description>::value;

    // block_dim_of
    template<class Description>
    using block_dim_of = commondx::block_dim_of<operator_type, Description>;
    template<class Description>
    inline constexpr dim3 block_dim_of_v = block_dim_of<Description>::value;

    // size_of
    template<class Description>
    struct size_of {
    private:
        static constexpr bool has_size = detail::has_operator<operator_type::size, Description>::value;
        static_assert(has_size, "Description does not have size defined");

    public:
        using value_type                   = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int m    = detail::get_t<operator_type::size, Description>::m;
        static constexpr unsigned int n    = detail::get_t<operator_type::size, Description>::n;
        static constexpr unsigned int k    = detail::get_t<operator_type::size, Description>::k;

        static constexpr value_type value = value_type {m, n, k};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> size_of_v = size_of<Description>::value;
    template<class Description>
    inline constexpr unsigned int size_of_v_m = size_of<Description>::m;
    template<class Description>
    inline constexpr unsigned int size_of_v_n = size_of<Description>::n;
    template<class Description>
    inline constexpr unsigned int size_of_v_k = size_of<Description>::k;

    // function_of
    template<class Description>
    struct function_of {
    private:
        static constexpr bool has_function = detail::has_operator_v<operator_type::function, Description>;

    public:
        using value_type = function;

        static constexpr value_type value = detail::get_or_default_t<operator_type::function, Description, detail::default_function_operator>::value;
    };

    template<class Description>
    inline constexpr function function_of_v = function_of<Description>::value;

    // fill_mode_of
    template<class Description>
    struct fill_mode_of {
        using value_type                  = fill_mode;
        static constexpr value_type value = detail::get_or_default_t<operator_type::fill_mode, Description, detail::default_fill_mode_operator>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr fill_mode fill_mode_of<Description>::value;
    template<class Description>
    inline constexpr fill_mode fill_mode_of_v = fill_mode_of<Description>::value;

    // arrangement_of
    template<class Description>
    struct arrangement_of {
    private:
        using arrangement_type  = detail::get_or_default_t<operator_type::arrangement, Description, detail::default_arrangement_operator>;
    public:
        using value_type               = COMMONDX_STL_NAMESPACE::tuple<arrangement, arrangement, arrangement>;
        static constexpr arrangement a = arrangement_type::a;
        static constexpr arrangement b = arrangement_type::b;
        static constexpr arrangement c = arrangement_type::c;
        static constexpr value_type value = value_type {a, b, c};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<arrangement, arrangement, arrangement> arrangement_of_v = arrangement_of<Description>::value;
    template<class Description>
    inline constexpr arrangement arrangement_of_v_a = arrangement_of<Description>::a;
    template<class Description>
    inline constexpr arrangement arrangement_of_v_b = arrangement_of<Description>::b;
    template<class Description>
    inline constexpr arrangement arrangement_of_v_c = arrangement_of<Description>::c;

    // transpose_mode_of
    template<class Description>
    struct transpose_mode_of {
        using value_type                  = transpose;
        static constexpr value_type value = detail::get_or_default_t<operator_type::transpose, Description, detail::default_transpose_operator>::value;
    };

    template<class Description>
    inline constexpr transpose transpose_mode_of_v = transpose_mode_of<Description>::value;

    // side_of
    template<class Description>
    struct side_of {
        using value_type                  = side;
        static constexpr value_type value = detail::get_or_default_t<operator_type::side, Description, Side<side::left>>::value;
    };

    template<class Description>
    inline constexpr side side_of_v = side_of<Description>::value;

    // diag_of
    template<class Description>
    struct diag_of {
        using value_type                  = diag;
        static constexpr value_type value = detail::get_or_default_t<operator_type::diag, Description, Diag<diag::non_unit>>::value;
    };

    template<class Description>
    inline constexpr diag diag_of_v = diag_of<Description>::value;

    // job_of
    template<class Description>
    struct job_of {
    private:
        using job_type = detail::get_or_default_t<operator_type::job, Description, Job<job::no_vectors>>;
    public:
        using value_type = COMMONDX_STL_NAMESPACE::tuple<job, job>;
        static constexpr job jobu = job_type::jobu;
        static constexpr job jobvt = job_type::jobvt;
        static constexpr value_type value = value_type {jobu, jobvt};
    };

    template<class Description>
    inline constexpr job job_of_v = job_of<Description>::jobu;
    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<job, job> jobs_of_v = job_of<Description>::value;
    template<class Description>
    inline constexpr job jobu_of_v = job_of<Description>::jobu;
    template<class Description>
    inline constexpr job jobvt_of_v = job_of<Description>::jobvt;

    // eig_type_of
    template<class Description>
    struct eig_type_of {
        using value_type                  = int;
        static constexpr value_type value = detail::get_or_default_t<operator_type::eig_type, Description, detail::default_eig_type_operator>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr int eig_type_of_v = eig_type_of<Description>::value;

    // leading dimension of
    namespace detail {
        template<unsigned int M, unsigned int N, unsigned int K, arrangement Arr, arrangement Brr, arrangement Crr, function Func, side Side, job Jobu, job Jobvt>
        struct default_leading_dimension {
            static constexpr bool is_function_unmq = is_unmq<Func>::value;
            static constexpr bool is_function_trsm = is_trsm<Func>::value;
            static constexpr bool is_function_gtsv = is_gtsv_no_pivot<Func>::value;

            // Regular leading dimension calculations
            static constexpr unsigned int lda_regular = (Arr == arrangement::col_major) ? M : N;
            static constexpr unsigned int ldb_regular = (Brr == arrangement::col_major) ? const_max(M, N) : K;

            // GTSV_NO_PIVOT specific leading dimension calculations
            static constexpr unsigned int lda_gtsv = M; // ignored
            static constexpr unsigned int ldb_gtsv = (Brr == arrangement::col_major) ? M : K;

            // UNMQ specific leading dimension calculations
            static constexpr unsigned int lda_unmq = (Func == function::unmqr) ?
                ((Arr == arrangement::col_major) ? 
                    ((Side == side::left) ? M : N) : K) :
                ((Arr == arrangement::col_major) ? K :
                    ((Side == side::left) ? M : N));
            static constexpr unsigned int ldb_unmq = (Brr == arrangement::col_major) ? M : N;

            // TRSM specific leading dimension calculations
            static constexpr unsigned int lda_trsm = (Side == side::left) ? M : N;
            static constexpr unsigned int ldb_trsm = (Brr == arrangement::col_major) ? M : N;

            // BDSVD specific leading dimension calculations for U and VT. lda and ldb are ignored for no_vectors
            static constexpr unsigned int lda_bdsvd = (Jobu != job::no_vectors) ? M : 0;
            static constexpr unsigned int ldb_bdsvd = (Jobvt != job::no_vectors) ? M : 0;

            // GESVD specific leading dimensions for U and VT
            static constexpr unsigned int act_m_u = M;
            static constexpr unsigned int act_n_u = Jobu == job::all_vectors ? M : const_min(M, N);
            static constexpr unsigned int act_m_vt = Jobvt == job::all_vectors ? N : const_min(M, N);
            static constexpr unsigned int act_n_vt = N;  
            static constexpr unsigned int ldb_gesvd = (Jobu != job::no_vectors && Jobu != job::overwrite_vectors) ? (Brr == arrangement::col_major ? act_m_u : act_n_u) : 0;
            static constexpr unsigned int ldc_gesvd = (Jobvt != job::no_vectors && Jobvt != job::overwrite_vectors) ? (Crr == arrangement::col_major ? act_m_vt : act_n_vt) : 0;

            // HEGV/HEGST specific leading dimensions for B
            static constexpr unsigned int ldb_hegv = M;

            // Choose leading dimensions
            static constexpr unsigned int lda =    is_function_unmq ?        lda_unmq
                                                : (is_function_trsm ?        lda_trsm
                                                : (is_function_gtsv ?        lda_gtsv
                                                : (Func == function::bdsvd ? lda_bdsvd
                                                : /*default:*/               lda_regular)));
            static constexpr unsigned int ldb =    is_function_unmq ?        ldb_unmq
                                                : (is_function_trsm ?        ldb_trsm
                                                : (is_function_gtsv ?        ldb_gtsv
                                                : (Func == function::bdsvd ? ldb_bdsvd
                                                : (Func == function::gesvd ? ldb_gesvd
                                                : (Func == function::hegst ? ldb_hegv
                                                : (Func == function::hegv  ? ldb_hegv
                                                : /*default:*/               ldb_regular))))));
            static constexpr unsigned int ldc =    Func == function::gesvd ? ldc_gesvd
                                                : /*default:*/ 0;

            using type = LeadingDimension<lda, ldb, ldc>;
        };

        template<unsigned int M, unsigned int N, unsigned int K, arrangement Arr, arrangement Brr, arrangement Crr, function Func, side Side, job Jobu, job Jobvt>
        using default_leading_dimension_t = typename default_leading_dimension<M, N, K, Arr, Brr, Crr, Func, Side, Jobu, Jobvt>::type;
    } // namespace detail

    template<class Description>
    struct leading_dimension_of {
    private:
        using this_size        = detail::get_or_default_t<operator_type::size, Description, Size<1, 1, 1>>;
        using this_arrangement = detail::get_or_default_t<operator_type::arrangement, Description, detail::default_arrangement_operator>;
        using this_function    = detail::get_or_default_t<operator_type::function, Description, detail::default_function_operator>;
        using this_side        = detail::get_or_default_t<operator_type::side, Description, detail::default_side_operator>;
        using this_job         = detail::get_or_default_t<operator_type::job, Description, detail::default_job_operator>;

        using default_ld = detail::default_leading_dimension_t<this_size::m,
                                                               this_size::n,
                                                               this_size::k,
                                                               this_arrangement::a,
                                                               this_arrangement::b,
                                                               this_arrangement::c,
                                                               this_function::value,
                                                               this_side::value,
                                                               this_job::jobu,
                                                               this_job::jobvt>;

    public:
        using value_type = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int a = detail::get_or_default_t<operator_type::leading_dimension, Description, default_ld>::a;
        static constexpr unsigned int b = detail::get_or_default_t<operator_type::leading_dimension, Description, default_ld>::b;
        static constexpr unsigned int c = detail::get_or_default_t<operator_type::leading_dimension, Description, default_ld>::c;
        static constexpr value_type value = value_type{a, b, c};
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> leading_dimension_of_v = leading_dimension_of<Description>::value;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_v_a = leading_dimension_of<Description>::a;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_v_b = leading_dimension_of<Description>::b;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_v_c = leading_dimension_of<Description>::c;

    // --------------------------
    // Matrix size calculations
    // --------------------------
    template<class Description>
    struct matrix_size_of {
    private:
        using this_size = detail::get_or_default_t<operator_type::size, Description, Size<1, 1, 1>>;
        using this_arrangement = detail::get_or_default_t<operator_type::arrangement, Description, detail::default_arrangement_operator>;
        using this_function = detail::get_or_default_t<operator_type::function, Description, detail::default_function_operator>;
        using this_side        = detail::get_or_default_t<operator_type::side, Description, detail::default_side_operator>;
        using this_job = detail::get_or_default_t<operator_type::job, Description, detail::default_job_operator>;

        static constexpr bool is_function_unmq          = detail::is_unmq<this_function::value>::value;
        static constexpr bool has_both_a_and_b_matrices = detail::has_both_a_and_b_matrices<this_function::value>::value;
        static constexpr bool is_function_qr            = detail::is_qr<this_function::value>::value;
        static constexpr bool is_function_lu_pp         = detail::is_lu_partial_pivot<this_function::value>::value;
        static constexpr bool is_function_gtsv_no_pivot = detail::is_gtsv_no_pivot<this_function::value>::value;
        static constexpr bool is_function_trsm          = detail::is_trsm<this_function::value>::value;
        static constexpr bool is_function_htev          = (this_function::value == function::htev);
        static constexpr bool is_function_heev          = (this_function::value == function::heev);
        static constexpr bool is_function_hegv          = (this_function::value == function::hegv);
        static constexpr bool is_function_bdsvd         = (this_function::value == function::bdsvd);
        static constexpr bool is_function_gesvd         = (this_function::value == function::gesvd);

        // Regular matrix size calculations
        static constexpr unsigned int a_size_regular = leading_dimension_of_v_a<Description> * 
            (this_arrangement::a == arrangement::col_major ? this_size::n : this_size::m);
        static constexpr unsigned int b_size_regular = leading_dimension_of_v_b<Description> * 
            (this_arrangement::b == arrangement::col_major ? this_size::k : detail::const_max(this_size::m, this_size::n));

        // GTSV_NO_PIVOT specific matrix size calculations
        static constexpr unsigned int a_size_gtsv_no_pivot = (3 * this_size::m - 2);
        static constexpr unsigned int b_size_gtsv_no_pivot = leading_dimension_of_v_b<Description> *
            (this_arrangement::b == arrangement::col_major ? this_size::k : this_size::m);

        // UNMQ specific matrix size calculations
        static constexpr unsigned int am = (this_function::value == function::unmqr) ? 
            (this_side::value == side::left ? this_size::m : this_size::n) : this_size::k;
        static constexpr unsigned int an = (this_function::value == function::unmqr) ? 
            this_size::k : (this_side::value == side::left ? this_size::m : this_size::n);
        static constexpr unsigned int a_size_unmq = leading_dimension_of_v_a<Description> * 
            (this_arrangement::a == arrangement::col_major ? an : am);
        static constexpr unsigned int b_size_unmq = leading_dimension_of_v_b<Description> * 
            (this_arrangement::b == arrangement::col_major ? this_size::n : this_size::m);
        
        // TRSM specific matrix size calculations
        static constexpr unsigned int a_size_trsm = leading_dimension_of_v_a<Description> * 
            (this_side::value == side::left ? this_size::m : this_size::n);
        static constexpr unsigned int b_size_trsm = leading_dimension_of_v_b<Description> * 
            (this_arrangement::b == arrangement::col_major ? this_size::n : this_size::m);

        // HTEV specific matrix size calculations
        static constexpr unsigned int a_size_htev_bdsvd = (this_size::m * 2 - 1);
        static constexpr unsigned int b_size_htev = this_job::jobu == job::no_vectors ? 0 : leading_dimension_of_v_a<Description> * this_size::m;

        // HEGV specific matrix size calculations
        static constexpr unsigned int b_size_hegv = leading_dimension_of_v_b<Description> * this_size::m;

        // BDSVD specific matrix size calculations
        static constexpr unsigned int b_size_bdsvd = this_job::jobu == job::no_vectors ? 0 : leading_dimension_of_v_a<Description> * this_size::m;
        static constexpr unsigned int c_size_bdsvd = this_job::jobvt == job::no_vectors ? 0 : leading_dimension_of_v_b<Description> * this_size::m;

        // GESVD specific matrix size calculations
        static constexpr unsigned int m_u          = this_size::m;
        static constexpr unsigned int n_u          = this_job::jobu == job::all_vectors ? this_size::m : detail::const_min(this_size::m, this_size::n);
        static constexpr unsigned int m_vt         = this_job::jobvt == job::all_vectors ? this_size::n : detail::const_min(this_size::m, this_size::n);
        static constexpr unsigned int n_vt         = this_size::n;

        static constexpr unsigned int b_size_gesvd = (this_job::jobu == job::no_vectors || this_job::jobu == job::overwrite_vectors) ? 0 : leading_dimension_of_v_b<Description> * (this_arrangement::b == arrangement::col_major ? n_u : m_u);
        static constexpr unsigned int c_size_gesvd = (this_job::jobvt == job::no_vectors || this_job::jobvt == job::overwrite_vectors) ? 0 : leading_dimension_of_v_c<Description> * (this_arrangement::c == arrangement::col_major ? n_vt : m_vt);

        static constexpr unsigned int extra_size_qr = is_function_unmq ? this_size::k : detail::const_min(this_size::m, this_size::n);
        static constexpr unsigned int extra_size_heev = this_size::m;
        static constexpr unsigned int extra_bytes_lu_pp = sizeof(int) * detail::const_min(this_size::m, this_size::n);
        static constexpr unsigned int extra_bytes_heev = sizeof(typename precision_of<Description>::a_type) * this_size::m * 2; // lambda and workspace
        static constexpr unsigned int extra_bytes_gesvd = sizeof(typename precision_of<Description>::a_type) * this_size::m * 3; // sigma or lambda and workspace

    public:
        using value_type = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int a_size =   is_function_unmq          ? a_size_unmq
                                              : (is_function_trsm          ? a_size_trsm
                                              : (is_function_htev          ? a_size_htev_bdsvd
                                              : (is_function_bdsvd         ? a_size_htev_bdsvd
                                              : (is_function_gtsv_no_pivot ? a_size_gtsv_no_pivot
                                              : /*default*/                  a_size_regular))));
        static constexpr unsigned int b_size =   is_function_unmq          ? b_size_unmq
                                              : (is_function_trsm          ? b_size_trsm
                                              : (is_function_gtsv_no_pivot ? b_size_gtsv_no_pivot
                                              : (is_function_htev          ? b_size_htev
                                              : (is_function_hegv          ? b_size_hegv
                                              : (is_function_bdsvd         ? b_size_bdsvd
                                              : (is_function_gesvd         ? b_size_gesvd
                                              : (has_both_a_and_b_matrices ? b_size_regular
                                              : /*default, no B*/            0)))))));
        static constexpr unsigned int c_size = is_function_bdsvd ? c_size_bdsvd : (is_function_gesvd ? c_size_gesvd : 0);
        static constexpr unsigned int extra_size = is_function_qr ? extra_size_qr : ((is_function_heev || is_function_hegv || is_function_gesvd) ? extra_size_heev : 0);
        static constexpr unsigned int extra_bytes =   is_function_lu_pp ? extra_bytes_lu_pp
                                                   : (is_function_heev  ? extra_bytes_heev
                                                   : (is_function_hegv  ? extra_bytes_heev
                                                   : (is_function_gesvd ? extra_bytes_gesvd
                                                   : /*default*/          0)));

        static constexpr value_type value = value_type{a_size, b_size, c_size, extra_size, extra_bytes};
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int> matrix_size_of_v = matrix_size_of<Description>::value;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_a = matrix_size_of<Description>::a_size;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_b = matrix_size_of<Description>::b_size;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_c = matrix_size_of<Description>::c_size;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_extra_size = matrix_size_of<Description>::extra_size;
    template<class Description>
    inline constexpr unsigned int matrix_size_of_v_extra_bytes = matrix_size_of<Description>::extra_bytes;

    // --------------------------
    // General Description traits
    // --------------------------

    template<class Description>
    using is_solver = commondx::is_dx_expression<Description>;

    template<class Description>
    inline constexpr bool is_solver_v = is_solver<Description>::value;

    template<class Description>
    using is_solver_execution = commondx::is_dx_execution_expression<operator_type, Description>;

    template<class Description>
    inline constexpr bool is_solver_execution_v = is_solver_execution<Description>::value;

    template<class Description>
    using is_complete_solver = commondx::is_complete_dx_expression<Description, detail::is_complete_description>;

    template<class Description>
    inline constexpr bool is_complete_solver_v = is_complete_solver<Description>::value;

    template<class Description>
    using is_complete_solver_execution = commondx::is_complete_dx_execution_expression<operator_type, Description, detail::is_complete_execution_description>;

    template<class Description>
    inline constexpr bool is_complete_solver_execution_v = is_complete_solver_execution<Description>::value;

    template<class Description>
    using extract_solver_description = commondx::extract_dx_description<detail::solver_description, Description, operator_type>;

    template<class Description>
    using extract_solver_description_t = typename extract_solver_description<Description>::type;

} // namespace cusolverdx

#endif // CUSOLVERDX_TRAITS_SOLVER_TRAITS_HPP
