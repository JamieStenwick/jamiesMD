// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_BLAS_TRAITS_TRAITS_HPP
#define CUBLASDX_BLAS_TRAITS_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "commondx/detail/stl/cstdint.hpp"

#include "cublasdx/detail/blas_description_fd.hpp"
#include "cublasdx/operators.hpp"
#include "cublasdx/types.hpp"
#include "cublasdx/traits/detail/description_traits.hpp"

#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/dx_traits.hpp"
#include "commondx/traits/numeric_traits.hpp"
#include "cute/int_tuple.hpp"

namespace cublasdx {
    namespace detail {
        template<commondx::data_type data_type, class Precision>
        struct map_value_type {
            using a_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex),
                                                                 complex<typename Precision::a_type>,
                                                                 typename Precision::a_type>;
            using b_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex),
                                                                 complex<typename Precision::b_type>,
                                                                 typename Precision::b_type>;
            using c_type = COMMONDX_STL_NAMESPACE::conditional_t<(data_type == commondx::data_type::complex),
                                                                 complex<typename Precision::c_type>,
                                                                 typename Precision::c_type>;
        };
    } // namespace detail

    // precision_of
    template<class Description>
    struct precision_of {
    private:
        using description = commondx::detail::get_or_default_t<operator_type,
                                                               operator_type::precision,
                                                               Description,
                                                               detail::default_blas_precision_operator>;

    public:
        using a_type = typename description::a_type;
        using b_type = typename description::b_type;
        using c_type = typename description::c_type;
    };

    template<class Description>
    using precision_of_a_t = typename precision_of<Description>::a_type;

    template<class Description>
    using precision_of_b_t = typename precision_of<Description>::b_type;

    template<class Description>
    using precision_of_c_t = typename precision_of<Description>::c_type;

    // type_of
    template<class Description>
    using type_of = commondx::data_type_of<operator_type, Description, detail::default_blas_type_operator>;

    template<class Description>
    inline constexpr type type_of_v = type_of<Description>::value;

    // sm_of
    template<class Description>
    struct sm_of {
    private:
        static constexpr bool has_sm = detail::has_operator<operator_type::sm, Description>::value;
        static_assert(has_sm, "Description does not have size defined");

    public:
        static constexpr unsigned    value    = detail::get_t<operator_type::sm, Description>::value;
        static constexpr sm_modifier modifier = detail::get_t<operator_type::sm, Description>::modifier;
    };

    template<class Description>
    inline constexpr unsigned int sm_of_v = sm_of<Description>::value;

    template<class Description>
    using is_block = detail::has_operator<operator_type::block, Description>;

    template<class Description>
    inline constexpr bool is_block_v = is_block<Description>::value;

    template<class Description>
    using is_thread = detail::has_operator<operator_type::thread, Description>;

    template<class Description>
    inline constexpr bool is_thread_v = is_thread<Description>::value;

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
        using value_type                = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int m = detail::get_t<operator_type::size, Description>::m;
        static constexpr unsigned int n = detail::get_t<operator_type::size, Description>::n;
        static constexpr unsigned int k = detail::get_t<operator_type::size, Description>::k;

        static constexpr value_type value = value_type {m, n, k};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> size_of_v =
        size_of<Description>::value;

    template<class Description>
    inline constexpr unsigned int size_of_v_m = size_of<Description>::m;

    template<class Description>
    inline constexpr unsigned int size_of_v_n = size_of<Description>::n;

    template<class Description>
    inline constexpr unsigned int size_of_v_k = size_of<Description>::k;

    // function_of
    template<class Description>
    struct function_of {
        using value_type                  = function;
        static constexpr value_type value = detail::get_t<operator_type::function, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }

        static_assert(detail::has_operator<operator_type::function, Description>::value, "Description does not have function defined");
    };

    template<class Description>
    constexpr function function_of<Description>::value;

    template<class Description>
    inline constexpr function function_of_v = function_of<Description>::value;

    // alignment_of
    template<class Description>
    struct alignment_of {
    private:
        using abc_type_map      = detail::map_value_type<type_of_v<Description>,
                                                         Precision<typename precision_of<Description>::a_type,
                                                                   typename precision_of<Description>::b_type,
                                                                   typename precision_of<Description>::c_type>>;
        using default_alignment = Alignment<alignof(typename abc_type_map::a_type),
                                            alignof(typename abc_type_map::b_type),
                                            alignof(typename abc_type_map::c_type)>;

        using alignment_type = detail::get_or_default_t<operator_type::alignment, Description, default_alignment>;

    public:
        using value_type                = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int a = alignment_type::a;
        static constexpr unsigned int b = alignment_type::b;
        static constexpr unsigned int c = alignment_type::c;

        static constexpr value_type value = value_type {a, b, c};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> alignment_of_v =
        alignment_of<Description>::value;

    template<class Description>
    inline constexpr unsigned int alignment_of_v_a = alignment_of<Description>::a;

    template<class Description>
    inline constexpr unsigned int alignment_of_v_b = alignment_of<Description>::b;

    template<class Description>
    inline constexpr unsigned int alignment_of_v_c = alignment_of<Description>::c;

    // transpose_mode_of
    namespace detail {
        template<class T>
        struct convert_to_transpose_mode;

        template<arrangement AOrder, arrangement BOrder, arrangement COrder>
        struct convert_to_transpose_mode<Arrangement<AOrder, BOrder, COrder>> {
            static constexpr transpose_mode a =
                (AOrder == arrangement::col_major) ? transpose_mode::non_transposed : transpose_mode::transposed;
            static constexpr transpose_mode b =
                (BOrder == arrangement::col_major) ? transpose_mode::non_transposed : transpose_mode::transposed;
            static constexpr transpose_mode c = transpose_mode::non_transposed;
            using type                        = TransposeMode<a, b>;
        };

        template<class T>
        struct convert_to_arrangement;

        template<transpose_mode A, transpose_mode B>
        struct convert_to_arrangement<TransposeMode<A, B>> {
            static constexpr arrangement a =
                (A == transpose_mode::non_transposed) ? arrangement::col_major : arrangement::row_major;
            static constexpr arrangement b =
                (B == transpose_mode::non_transposed) ? arrangement::col_major : arrangement::row_major;
            static constexpr arrangement c = arrangement::col_major;
            using type                     = Arrangement<a, b, c>;
        };
    } // namespace detail

    template<class Description>
    struct transpose_mode_of {
    private:
        static constexpr bool has_arrangement = detail::has_operator<operator_type::arrangement, Description>::value;
        using default_value =
            COMMONDX_STL_NAMESPACE::conditional_t<has_arrangement,
                                                  typename detail::convert_to_transpose_mode<detail::get_or_default_t<
                                                      operator_type::arrangement,
                                                      Description,
                                                      detail::default_blas_arrangement_operator>>::type,
                                                  detail::default_blas_transpose_mode_operator>;

    public:
        using value_type = transpose_mode;

        static constexpr value_type a_transpose_mode =
            detail::get_or_default_t<operator_type::transpose_mode, Description, default_value>::a_transpose_mode;
        static constexpr value_type b_transpose_mode =
            detail::get_or_default_t<operator_type::transpose_mode, Description, default_value>::b_transpose_mode;
    };

    template<class Description>
    inline constexpr transpose_mode transpose_mode_of_a = transpose_mode_of<Description>::a_transpose_mode;
    template<class Description>
    inline constexpr transpose_mode transpose_mode_of_b = transpose_mode_of<Description>::b_transpose_mode;

    // arrangement_of
    template<class Description>
    struct arrangement_of {
    private:
        static constexpr bool has_transpose_mode =
            detail::has_operator<operator_type::transpose_mode, Description>::value;
        using default_value =
            COMMONDX_STL_NAMESPACE::conditional_t<has_transpose_mode,
                                                  typename detail::convert_to_arrangement<detail::get_or_default_t<
                                                      operator_type::transpose_mode,
                                                      Description,
                                                      detail::default_blas_transpose_mode_operator>>::type,
                                                  detail::default_blas_arrangement_operator>;
        using arrangement_type = detail::get_or_default_t<operator_type::arrangement, Description, default_value>;

    public:
        using value_type               = COMMONDX_STL_NAMESPACE::tuple<arrangement, arrangement, arrangement>;
        static constexpr arrangement a = arrangement_type::a;
        static constexpr arrangement b = arrangement_type::b;
        static constexpr arrangement c = arrangement_type::c;

        static constexpr value_type value = value_type {a, b, c};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<arrangement, arrangement, arrangement> arrangement_of_v =
        arrangement_of<Description>::value;

    template<class Description>
    inline constexpr arrangement arrangement_of_v_a = arrangement_of<Description>::a;

    template<class Description>
    inline constexpr arrangement arrangement_of_v_b = arrangement_of<Description>::b;

    template<class Description>
    inline constexpr arrangement arrangement_of_v_c = arrangement_of<Description>::c;

    template<class Description>
    inline constexpr bool has_static_block_dim_v =
        detail::has_operator<operator_type::static_block_dim, Description>::value;

    namespace detail {
        template<class Representation>
        constexpr int required_mantissa_bits_to_emulation_slices(unsigned int const required_mantissa_bits) {
            // Account for the hidden significand bit.
            return cute::ceil_div(required_mantissa_bits + 1, cute::sizeof_bits_v<Representation>);
        }

        template<class T>
        struct default_required_mantissa_bits {
            static constexpr unsigned int value = 0;
        };

        template<>
        struct default_required_mantissa_bits<float> {
            static constexpr unsigned int value = 23;
        };

        template<>
        struct default_required_mantissa_bits<double> {
            static constexpr unsigned int value = 52;
        };
    }

    template<class Description>
    struct required_mantissa_bits_of {
        static constexpr bool is_emulated =
            detail::has_operator<operator_type::required_mantissa_bits, Description>::value;
        using default_required_mantissa_bits_t = COMMONDX_STL_NAMESPACE::integral_constant<
            unsigned int,
            detail::default_required_mantissa_bits<precision_of_a_t<Description>>::value>;
        static constexpr unsigned int value = detail::get_or_default_t<operator_type::required_mantissa_bits,
                                                              Description,
                                                              default_required_mantissa_bits_t>::value;
    };

    template<class Description>
    inline constexpr bool has_required_mantissa_bits_v = required_mantissa_bits_of<Description>::is_emulated;

    template<class Description>
    inline constexpr unsigned int required_mantissa_bits_of_v = required_mantissa_bits_of<Description>::value;

    template<class Description>
    struct emulation_slice_of {
        static constexpr bool is_enabled = required_mantissa_bits_of<Description>::is_emulated;
        static constexpr int value =
            detail::required_mantissa_bits_to_emulation_slices<COMMONDX_STL_NAMESPACE::int8_t>(
                required_mantissa_bits_of_v<Description>);
    };

    template<class Description>
    inline constexpr bool has_emulation_slice_v = emulation_slice_of<Description>::is_enabled;

    template<class Description>
    inline constexpr int emulation_slice_of_v = emulation_slice_of<Description>::value;

    // leading_dimension_of
    namespace detail {
        template<unsigned int M, unsigned int N, unsigned int K, arrangement AArr, arrangement BArr, arrangement CArr>
        struct default_leading_dimension {
            using type = LeadingDimension<((AArr == arrangement::col_major) ? M : K),
                                          ((BArr == arrangement::col_major) ? K : N),
                                          ((CArr == arrangement::col_major) ? M : N)>;
        };

        template<unsigned int M, unsigned int N, unsigned int K, arrangement AArr, arrangement BArr, arrangement CArr>
        using default_leading_dimension_t = typename default_leading_dimension<M, N, K, AArr, BArr, CArr>::type;
    } // namespace detail

    template<class Description>
    struct leading_dimension_of {
    private:
        static constexpr bool has_size = detail::has_operator<operator_type::size, Description>::value;
        static constexpr bool has_ld   = detail::has_operator<operator_type::ld, Description>::value;
        static_assert(has_size || has_ld, "Description does not have size nor leading dimensions defined");

        static constexpr bool has_tm = detail::has_operator<operator_type::transpose_mode, Description>::value;
        using default_arrangement =
            COMMONDX_STL_NAMESPACE::conditional_t<has_tm,
                                                  typename detail::convert_to_arrangement<detail::get_or_default_t<
                                                      operator_type::transpose_mode,
                                                      Description,
                                                      detail::default_blas_transpose_mode_operator>>::type,
                                                  detail::default_blas_arrangement_operator>;
        using description_arrangement =
            detail::get_or_default_t<operator_type::arrangement, Description, default_arrangement>;

        static constexpr auto arrangement_a = description_arrangement::a;
        static constexpr auto arrangement_b = description_arrangement::b;
        static constexpr auto arrangement_c = description_arrangement::c;

        static constexpr auto m = size_of<Description>::m;
        static constexpr auto n = size_of<Description>::n;
        static constexpr auto k = size_of<Description>::k;

        static constexpr function fun = detail::get_or_default_t<operator_type::function, Description, Function<function::MM>>::value;
        static constexpr bool is_mm = (fun == function::MM);
        static constexpr bool is_left = detail::get_or_default_t<operator_type::side, Description, Side<side::left>>::value == side::left;
        static constexpr int trsm_default_lda = is_left ? m : n;
        static constexpr int trsm_default_ldb = (arrangement_b == arrangement::col_major) ? m : n;

        using default_gemm_ld = detail::default_leading_dimension_t<m, n, k, arrangement_a, arrangement_b, arrangement_c>;
        using default_trsm_ld = LeadingDimension<trsm_default_lda, trsm_default_ldb>;

        using default_ld = COMMONDX_STL_NAMESPACE::conditional_t<is_mm, default_gemm_ld, default_trsm_ld>;
    public:
        using value_type                = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr unsigned int a = detail::get_or_default_t<operator_type::ld, Description, default_ld>::a;
        static constexpr unsigned int b = detail::get_or_default_t<operator_type::ld, Description, default_ld>::b;
        static constexpr unsigned int c = detail::get_or_default_t<operator_type::ld, Description, default_ld>::c;

        static constexpr value_type value = value_type {a, b, c};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> leading_dimension_of_v =
        leading_dimension_of<Description>::value;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_v_a = leading_dimension_of<Description>::a;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_v_b = leading_dimension_of<Description>::b;
    template<class Description>
    inline constexpr unsigned int leading_dimension_of_v_c = leading_dimension_of<Description>::c;

    template<class Description>
    struct batches_per_block_of {
        using value_type = unsigned int;
        static constexpr value_type value = detail::get_or_default_t<operator_type::batches_per_block, Description, BatchesPerBlock<1>>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr unsigned int batches_per_block_of_v = batches_per_block_of<Description>::value;

    template<class Description>
    struct fill_mode_of {
        using value_type                  = fill_mode;
        static constexpr value_type value = detail::get_or_default_t<operator_type::fill_mode, Description, FillMode<fill_mode::lower>>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr fill_mode fill_mode_of_v = fill_mode_of<Description>::value;

    template<class Description>
    struct diag_of {
        using value_type                  = diag;
        static constexpr value_type value = detail::get_or_default_t<operator_type::diag, Description, Diag<diag::non_unit>>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr diag diag_of_v = diag_of<Description>::value;

    template<class Description>
    struct side_of {
        using value_type                  = side;
        static constexpr value_type value = detail::get_or_default_t<operator_type::side, Description, Side<side::left>>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr side side_of_v = side_of<Description>::value;

    // is_blas
    template<class Description>
    using is_blas = commondx::is_dx_expression<Description>;

    template<class Description>
    inline constexpr bool is_blas_v = commondx::is_dx_expression<Description>::value;

    // is_blas_execution
    template<class Description>
    using is_blas_execution = commondx::is_dx_execution_expression<operator_type, Description>;

    template<class Description>
    inline constexpr bool is_blas_execution_v = commondx::is_dx_execution_expression<operator_type, Description>::value;

    // is_complete_blas
    template<class Description>
    using is_complete_blas = commondx::is_complete_dx_expression<Description, detail::is_complete_description>;

    template<class Description>
    inline constexpr bool is_complete_blas_v =
        commondx::is_complete_dx_expression<Description, detail::is_complete_description>::value;

    // is_complete_blas_execution
    template<class Description>
    using is_complete_blas_execution =
        commondx::is_complete_dx_execution_expression<operator_type, Description, detail::is_complete_description>;

    template<class Description>
    inline constexpr bool is_complete_blas_execution_v = commondx::
        is_complete_dx_execution_expression<operator_type, Description, detail::is_complete_description>::value;

    // extract_blas_description
    template<class Description>
    using extract_blas_description =
        commondx::extract_dx_description<detail::blas_description, Description, operator_type>;

    template<class Description>
    using extract_blas_description_t = typename extract_blas_description<Description>::type;

    // suggested_leading_dimension_of
    namespace detail {
        template<typename T, unsigned size>
        static constexpr unsigned pad_if_needed() {
            constexpr unsigned size_B      = sizeof(T) * size;
            constexpr bool     add_padding = (size_B >= 128 && size_B % 128 == 0);
            return add_padding ? size + 16 / sizeof(T) : size;
        }

        // forward declaration
        template<class PA,
                 class PB,
                 class PC,
                 class Alignment,
                 cublasdx::type Type,
                 arrangement    AArr,
                 arrangement    BArr,
                 arrangement    CArr,
                 class Size,
                 class LD,
                 unsigned int Arch>
        struct is_supported_restrict_impl;

        template<class Size, class Arrangement, class Precision, class Alignment, type Type, unsigned int Architecture>
        struct suggested_leading_dimension_of_impl {
            static constexpr auto m             = Size::m;
            static constexpr auto n             = Size::n;
            static constexpr auto k             = Size::k;
            static constexpr auto a_arrangement = Arrangement::a;
            static constexpr auto b_arrangement = Arrangement::b;
            static constexpr auto c_arrangement = Arrangement::c;

            using a_value_type = typename map_value_type<Type, Precision>::a_type;
            using b_value_type = typename map_value_type<Type, Precision>::b_type;
            using c_value_type = typename map_value_type<Type, Precision>::c_type;

            static constexpr unsigned int best_lda = (a_arrangement == arrangement::col_major)
                                                         ? pad_if_needed<a_value_type, m>()
                                                         : pad_if_needed<a_value_type, k>();
            static constexpr unsigned int best_ldb = (b_arrangement == arrangement::col_major)
                                                         ? pad_if_needed<b_value_type, k>()
                                                         : pad_if_needed<b_value_type, n>();
            static constexpr unsigned int best_ldc = (c_arrangement == arrangement::col_major)
                                                         ? pad_if_needed<c_value_type, m>()
                                                         : pad_if_needed<c_value_type, n>();

            // Default leading dimensions
            using default_ld = default_leading_dimension_t<m, n, k, a_arrangement, b_arrangement, c_arrangement>;

            // Check if the operation is supported with LD(best_lda, best_ldb, best_ldc)
            using precision_a_t = typename Precision::a_type;
            using precision_b_t = typename Precision::b_type;
            using precision_c_t = typename Precision::c_type;
            static constexpr bool padding_supported =
                is_supported_restrict_impl<precision_a_t,
                                           precision_b_t,
                                           precision_c_t,
                                           Alignment,
                                           Type,
                                           a_arrangement,
                                           b_arrangement,
                                           c_arrangement,
                                           Size,
                                           LeadingDimension<best_lda, best_ldb, best_ldc>,
                                           Architecture>::rmem_value();

            // Use default or best leading dimensions
            static constexpr unsigned int lda = padding_supported ? best_lda : default_ld::a;
            static constexpr unsigned int ldb = padding_supported ? best_ldb : default_ld::b;
            static constexpr unsigned int ldc = padding_supported ? best_ldc : default_ld::c;
        };
    } // namespace detail

    template<class Description, unsigned int Architecture>
    struct suggested_leading_dimension_of {
        static_assert(detail::get_or_default_t<operator_type::function, Description, Function<function::MM>>::value == function::MM, "Suggested leading dimension is only supported for GEMM");
    private:
        static constexpr bool has_size = detail::has_operator<operator_type::size, Description>::value;
        static_assert(has_size, "Description does not have size defined");

        using blas_precision              = precision_of<Description>;
        static constexpr auto blas_type_v = type_of_v<Description>;

        static constexpr bool has_tm = detail::has_operator<operator_type::transpose_mode, Description>::value;
        using blas_arrangement       = COMMONDX_STL_NAMESPACE::conditional_t<
                  has_tm,
                  typename detail::convert_to_arrangement<
                      detail::get_or_default_t<operator_type::transpose_mode,
                                               Description,
                                               detail::default_blas_transpose_mode_operator>>::type,
                  detail::
                      get_or_default_t<operator_type::arrangement, Description, detail::default_blas_arrangement_operator>>;

        using suggested = detail::suggested_leading_dimension_of_impl<
            detail::get_t<operator_type::size, Description>,
            blas_arrangement,
            blas_precision,
            detail::get_or_default_t<operator_type::alignment,
                                     Description,
                                     Alignment<sizeof(typename blas_precision::a_type),
                                               sizeof(typename blas_precision::b_type),
                                               sizeof(typename blas_precision::c_type)>>,
            blas_type_v,
            Architecture>;

    public:
        static constexpr unsigned int lda = suggested::lda;
        static constexpr unsigned int ldb = suggested::ldb;
        static constexpr unsigned int ldc = suggested::ldc;
        // Suggested leading dimensions of A, B matrices that can provide a good performance
        using type = LeadingDimension<lda, ldb, ldc>;

        using value_type                  = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr value_type value = value_type {lda, ldb, ldc};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description, unsigned int Architecture>
    using suggested_leading_dimension_of_t = typename suggested_leading_dimension_of<Description, Architecture>::type;

    template<class Description, unsigned int Architecture>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>
        suggested_leading_dimension_of_v = suggested_leading_dimension_of<Description, Architecture>::value;

    template<class Description, unsigned int Architecture>
    inline constexpr unsigned int suggested_leading_dimension_of_v_a =
        suggested_leading_dimension_of<Description, Architecture>::lda;

    template<class Description, unsigned int Architecture>
    inline constexpr unsigned int suggested_leading_dimension_of_v_b =
        suggested_leading_dimension_of<Description, Architecture>::ldb;

    template<class Description, unsigned int Architecture>
    inline constexpr unsigned int suggested_leading_dimension_of_v_c =
        suggested_leading_dimension_of<Description, Architecture>::ldc;

    // suggested_alignment_of
    template<class Description>
    struct suggested_alignment_of {
        static constexpr unsigned int a_alignment = MaxAlignment::a;
        static constexpr unsigned int b_alignment = MaxAlignment::b;
        static constexpr unsigned int c_alignment = MaxAlignment::c;
        // Suggested alignments of A, B, C matrices that can provide a good performance
        using type = MaxAlignment;

        using value_type                  = COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int>;
        static constexpr value_type value = value_type {MaxAlignment::a, MaxAlignment::b, MaxAlignment::c};
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    using suggested_alignment_of_t = typename suggested_alignment_of<Description>::type;

    template<class Description>
    inline constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int, unsigned int> suggested_alignment_of_v =
        suggested_alignment_of<Description>::value;

    template<class Description>
    inline constexpr unsigned int suggested_alignment_of_v_a = suggested_alignment_of<Description>::a_alignment;

    template<class Description>
    inline constexpr unsigned int suggested_alignment_of_v_b = suggested_alignment_of<Description>::b_alignment;

    template<class Description>
    inline constexpr unsigned int suggested_alignment_of_v_c = suggested_alignment_of<Description>::c_alignment;
} // namespace cublasdx

#endif // CUBLASDX_BLAS_TRAITS_TRAITS_HPP
