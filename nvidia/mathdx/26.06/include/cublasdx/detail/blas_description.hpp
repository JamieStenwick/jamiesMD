// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP
#define CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/traits/detail/get.hpp"
#include "commondx/detail/expressions.hpp"

#include "cublasdx/operators.hpp"
#include "cublasdx/traits/detail/description_traits.hpp"
#include "cublasdx/detail/blas_checks.hpp"

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

namespace cublasdx {
    namespace detail {
        constexpr unsigned int calculate_matrix_size(unsigned int ld, unsigned int x, unsigned int y, arrangement arr) {
            const unsigned size_other = ((arr == arrangement::col_major) ? y : x);
            const unsigned size_ld    = ((arr == arrangement::col_major) ? x : y);
            return ld * (size_other - 1) + size_ld;
        }

        constexpr unsigned int calculate_matrix_size(unsigned int   ld,
                                                     unsigned int   x,
                                                     unsigned int   y,
                                                     transpose_mode tmode) {
            const unsigned size_other = ((tmode == N) ? y : x);
            const unsigned size_ld    = ((tmode == N) ? x : y);
            return ld * (size_other - 1) + size_ld;
        }

        template<size_t org_size, size_t alignment = 16>
        inline constexpr size_t aligned_size() {
            return ((org_size + alignment - 1) / alignment) * alignment;
        }
        template<size_t alignment = 16>
        inline constexpr size_t aligned_dynamic_size(const size_t org_size) {
            return ((org_size + alignment - 1) / alignment) * alignment;
        }

        template<class... Operators>
        class blas_operator_wrapper: public commondx::detail::description_expression
        {
        };

        template<class... Operators>
        class blas_description: public commondx::detail::description_expression
        {
            using description_type = blas_operator_wrapper<Operators...>;

        protected:
            /// ---- Traits

            // Size
            // * Default value: NONE
            // * If there is no size, then dummy size is (8, 8, 8). This is required value for M, N sized don't break.
            // * Values of has_size or is_complete should be checked before using this property.
            static constexpr bool has_size = has_operator<operator_type::size, description_type>::value;
            using dummy_default_blas_size  = Size<8, 8, 8>;
            using this_blas_size = get_or_default_t<operator_type::size, description_type, dummy_default_blas_size>;
            static constexpr auto this_blas_size_m_v = this_blas_size::m;
            static constexpr auto this_blas_size_n_v = this_blas_size::n;
            static constexpr auto this_blas_size_k_v = this_blas_size::k;

            // Type (real, complex)
            // * Default value: real
            using this_blas_type = get_or_default_t<operator_type::type, description_type, default_blas_type_operator>;
            static constexpr auto this_blas_type_v = this_blas_type::value;

            // Function
            // * Default value: NONE
            // * Dummy value: MM
            static constexpr bool has_function = has_operator<operator_type::function, description_type>::value;
            using dummy_default_blas_function  = Function<function::MM>;
            using this_blas_function =
                get_or_default_t<operator_type::function, description_type, dummy_default_blas_function>;
            static constexpr auto this_blas_function_v = this_blas_function::value;

            // Precision
            // * Default: A, B, C are all float
            static constexpr bool has_precision = has_operator<operator_type::precision, description_type>::value;
            using this_blas_precision =
                get_or_default_t<operator_type::precision, description_type, default_blas_precision_operator>;

            // SM
            // * Default value: NONE
            // * Dummy value: 750
            static constexpr bool has_sm = has_operator<operator_type::sm, description_type>::value;
            using dummy_default_blas_sm  = SM<750>;
            using this_blas_sm           = get_or_default_t<operator_type::sm, description_type, dummy_default_blas_sm>;
            static constexpr auto this_blas_sm_v          = this_blas_sm::value;
            static constexpr auto this_blas_sm_modifier_v = this_blas_sm::modifier;

#if (__GNUC__ < 8) // Bug workaround
        public:
#endif
            // Arrangement, TransposeMode
            static constexpr bool has_arrangement = has_operator<operator_type::arrangement, description_type>::value;
            static constexpr bool has_transpose_mode =
                has_operator<operator_type::transpose_mode, description_type>::value;

            // Arrangement
            // * Default value: col_major, col_major, col_major
            using default_blas_arrangement = COMMONDX_STL_NAMESPACE::conditional_t<
                has_transpose_mode,
                typename convert_to_arrangement<get_or_default_t<operator_type::transpose_mode,
                                                                 description_type,
                                                                 default_blas_transpose_mode_operator>>::type,
                default_blas_arrangement_operator>;
            using this_blas_arrangement =
                get_or_default_t<operator_type::arrangement, description_type, default_blas_arrangement>;
            static constexpr auto this_blas_arrangement_a = this_blas_arrangement::a;
            static constexpr auto this_blas_arrangement_b = this_blas_arrangement::b;
            static constexpr auto this_blas_arrangement_c = this_blas_arrangement::c;

            // Alignment
            static constexpr bool has_alignment = has_operator<operator_type::alignment, description_type>::value;
            using default_blas_alignment =
                Alignment<alignof(typename map_value_type<this_blas_type_v, this_blas_precision>::a_type),
                          alignof(typename map_value_type<this_blas_type_v, this_blas_precision>::b_type),
                          alignof(typename map_value_type<this_blas_type_v, this_blas_precision>::c_type)>;
            using this_blas_alignment =
                get_or_default_t<operator_type::alignment, description_type, default_blas_alignment>;
            static constexpr auto this_blas_alignment_a = this_blas_alignment::a;
            static constexpr auto this_blas_alignment_b = this_blas_alignment::b;
            static constexpr auto this_blas_alignment_c = this_blas_alignment::c;

            // Has overloaded MMA Tile
            static constexpr bool has_overloaded_tile =
                has_operator<operator_type::experimental_tile, description_type>::value;
            using this_blas_overloaded_tile =
                get_or_default_t<operator_type::experimental_tile, description_type, experimental::Tile<void, 0, 0>>;

#if (__GNUC__ < 8) // Bug workaround
        protected:
#endif

            // TransposeMode
            // * Default value: N, N
            using default_blas_transpose_mode = COMMONDX_STL_NAMESPACE::conditional_t<
                has_arrangement,
                typename convert_to_transpose_mode<
                    get_or_default_t<operator_type::arrangement, description_type, default_blas_arrangement_operator>>::
                    type,
                default_blas_transpose_mode_operator>;
            using this_blas_transpose_mode =
                get_or_default_t<operator_type::transpose_mode, description_type, default_blas_transpose_mode>;
            static constexpr auto this_blas_transpose_mode_a = this_blas_transpose_mode::a_transpose_mode;
            static constexpr auto this_blas_transpose_mode_b = this_blas_transpose_mode::b_transpose_mode;

            // LeadingDimension
            static constexpr bool         has_ld = has_operator<operator_type::ld, description_type>::value;
            static constexpr unsigned int default_lda =
                ((this_blas_arrangement_a == arrangement::col_major) ? this_blas_size_m_v : this_blas_size_k_v);
            static constexpr unsigned int default_ldb =
                ((this_blas_arrangement_b == arrangement::col_major) ? this_blas_size_k_v : this_blas_size_n_v);
            static constexpr unsigned int default_ldc =
                ((this_blas_arrangement_c == arrangement::col_major) ? this_blas_size_m_v : this_blas_size_n_v);

            using dummy_default_blas_ld = LeadingDimension<1, 1, 1>;
            static constexpr auto this_blas_lda =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::a : default_lda;
            static constexpr auto this_blas_ldb =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::b : default_ldb;
            static constexpr auto this_blas_ldc =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::c : default_ldc;
            using this_blas_ld = LeadingDimension<this_blas_lda, this_blas_ldb, this_blas_ldc>;

            // Number of real elements in each matrix (includes padding)
            static constexpr auto this_blas_a_size =
                calculate_matrix_size(this_blas_lda, this_blas_size_m_v, this_blas_size_k_v, this_blas_arrangement_a);
            static constexpr auto this_blas_b_size =
                calculate_matrix_size(this_blas_ldb, this_blas_size_k_v, this_blas_size_n_v, this_blas_arrangement_b);
            static constexpr auto this_blas_c_size =
                calculate_matrix_size(this_blas_ldc, this_blas_size_m_v, this_blas_size_n_v, this_blas_arrangement_c);

            // Has forced static blockdim size
            static constexpr bool has_static_block_dim =
                has_operator<operator_type::static_block_dim, description_type>::value;
            using this_blas_static_block_dim = COMMONDX_STL_NAMESPACE::integral_constant<bool, has_static_block_dim>;

            static constexpr bool has_streaming = has_operator<operator_type::streaming, description_type>::value;
            using this_blas_streaming           = COMMONDX_STL_NAMESPACE::integral_constant<bool, has_streaming>;
            static constexpr bool this_blas_streaming_v = this_blas_streaming::value;

            static constexpr bool has_with_pipeline =
                has_operator<operator_type::with_pipeline, description_type>::value;
            using this_blas_with_pipeline = COMMONDX_STL_NAMESPACE::integral_constant<bool, has_with_pipeline>;
            static constexpr bool this_blas_with_pipeline_v = this_blas_with_pipeline::value;

            static constexpr bool has_required_mantissa_bits =
                has_operator<operator_type::required_mantissa_bits, description_type>::value;
            using this_blas_required_mantissa_bits = get_or_default_t<operator_type::required_mantissa_bits,
                                                                      description_type,
                                                                      RequiredMantissaBits<1>>;
            static constexpr unsigned int this_blas_required_mantissa_bits_v = this_blas_required_mantissa_bits::value;

            // Side (for TRSM; default: left).
            static constexpr bool has_side = has_operator<operator_type::side, description_type>::value;
            using this_blas_side           = get_or_default_t<operator_type::side, description_type, Side<side::left>>;
            static constexpr auto this_blas_side_v = this_blas_side::value;

            // Diag (for TRSM; default: non_unit).
            static constexpr bool has_diag = has_operator<operator_type::diag, description_type>::value;
            using this_blas_diag           = get_or_default_t<operator_type::diag, description_type, Diag<diag::non_unit>>;
            static constexpr auto this_blas_diag_v = this_blas_diag::value;

            // FillMode (for TRSM; default: lower).
            static constexpr bool has_fill_mode = has_operator<operator_type::fill_mode, description_type>::value;
            using this_blas_fill_mode           = get_or_default_t<operator_type::fill_mode, description_type, FillMode<fill_mode::lower>>;
            static constexpr auto this_blas_fill_mode_v = this_blas_fill_mode::value;

            // TRSM-specific derived dimensions.
            // For TRSM: A is dim_axdim_a (square triangular matrix), B is MxN.
            static constexpr unsigned this_blas_trsm_dim_a =
                (this_blas_side_v == side::left) ? this_blas_size_m_v : this_blas_size_n_v;
            // Natural (minimum) leading dimensions for TRSM.
            static constexpr unsigned this_blas_trsm_natural_lda = this_blas_trsm_dim_a;
            static constexpr unsigned this_blas_trsm_natural_ldb =
                (this_blas_arrangement_b == arrangement::col_major) ? this_blas_size_m_v : this_blas_size_n_v;
            // Effective LDs: respect LeadingDimension<LDA, LDB, ...>() operator when present.
            static constexpr unsigned this_blas_trsm_lda =
                has_ld ? static_cast<unsigned>(this_blas_lda) : this_blas_trsm_natural_lda;
            static constexpr unsigned this_blas_trsm_ldb =
                has_ld ? static_cast<unsigned>(this_blas_ldb) : this_blas_trsm_natural_ldb;
            // Storage sizes per batch (including LD padding).
            static constexpr unsigned this_blas_trsm_a_size = this_blas_trsm_lda * this_blas_trsm_dim_a;
            static constexpr unsigned this_blas_trsm_b_size =
                this_blas_trsm_ldb *
                ((this_blas_arrangement_b == arrangement::col_major) ? this_blas_size_n_v : this_blas_size_m_v);

            // BatchesPerBlock - optional user override for TRSM BPB.
            static constexpr bool has_batches_per_block =
            has_operator<operator_type::batches_per_block, description_type>::value;
        
            static constexpr bool has_thread = has_operator<operator_type::thread, description_type>::value;
            static constexpr bool has_block = has_operator<operator_type::block, description_type>::value;
            static constexpr bool has_block_dim = has_operator<operator_type::block_dim, description_type>::value;

            // True if description is complete description
            static constexpr bool is_complete = is_complete_description<description_type>::value;

            // Experimental --> testing 
            static constexpr bool         has_tile = has_operator<operator_type::experimental_tile, description_type>::value;

            /// ---- Constraints

            // We can only have one of each option

            // Main operators
            static constexpr bool has_one_function =
                has_at_most_one_of<operator_type::function, description_type>::value;
            static constexpr bool has_one_precision =
                has_at_most_one_of<operator_type::precision, description_type>::value;
            static constexpr bool has_one_size = has_at_most_one_of<operator_type::size, description_type>::value;
            static constexpr bool has_one_sm   = has_at_most_one_of<operator_type::sm, description_type>::value;
            static constexpr bool has_one_type = has_at_most_one_of<operator_type::type, description_type>::value;
            static constexpr bool has_one_block_dim =
                has_at_most_one_of<operator_type::block_dim, description_type>::value;
            static constexpr bool has_one_side             = has_at_most_one_of<operator_type::side, description_type>::value;
            static constexpr bool has_one_diag             = has_at_most_one_of<operator_type::diag, description_type>::value;
            static constexpr bool has_one_fill_mode        = has_at_most_one_of<operator_type::fill_mode, description_type>::value;
            static constexpr bool has_one_batches_per_block =
                has_at_most_one_of<operator_type::batches_per_block, description_type>::value;
            static constexpr bool has_one_alignment =
                has_at_most_one_of<operator_type::alignment, description_type>::value;
            static constexpr bool has_one_ld = has_at_most_one_of<operator_type::ld, description_type>::value;
            static constexpr bool has_one_transpose_mode =
                has_at_most_one_of<operator_type::transpose_mode, description_type>::value;
            static constexpr bool has_one_arrangement =
                has_at_most_one_of<operator_type::arrangement, description_type>::value;

            // experimental
            static constexpr bool has_one_tile =
                has_at_most_one_of<operator_type::experimental_tile, description_type>::value;
            static constexpr bool has_one_static_block_dim =
                has_at_most_one_of<operator_type::static_block_dim, description_type>::value;
            static constexpr bool has_one_streaming =
                has_at_most_one_of<operator_type::streaming, description_type>::value;
            static constexpr bool has_one_with_pipeline =
                has_at_most_one_of<operator_type::with_pipeline, description_type>::value;
            static constexpr bool has_one_required_mantissa_bits =
                has_at_most_one_of<operator_type::required_mantissa_bits, description_type>::value;

            static_assert(has_one_function, "Can't create blas function with two Function<> expressions");
            static_assert(has_one_precision, "Can't create blas function with two Precision<> expressions");
            static_assert(has_one_size, "Can't create blas function with two Size<> expressions");
            static_assert(has_one_sm, "Can't create blas function with two SM<> expressions");
            static_assert(has_one_type, "Can't create blas function with two Type<> expressions");
            static_assert(has_one_block_dim, "Can't create blas function with two BlockDim<> expressions");
            static_assert(has_one_side,              "Can't create blas function with two Side<> expressions");
            static_assert(has_one_diag,              "Can't create blas function with two Diag<> expressions");
            static_assert(has_one_fill_mode,         "Can't create blas function with two FillMode<> expressions");
            static_assert(has_one_batches_per_block, "Can't create blas function with two BatchesPerBlock<> expressions");
            static_assert(has_one_alignment, "Can't create blas function with two Alignment<> expressions");
            static_assert(has_one_ld, "Can't create blas function with two LeadingDimension<> expressions");
            static_assert(has_one_transpose_mode, "Can't create blas function with two TransposeMode<> expressions");
            static_assert(has_one_arrangement, "Can't create blas function with two Arrangement<> expressions");
            // experimental
            static_assert(has_one_tile, "Can't create blas function with two Tile<> expressions");
            static_assert(has_one_static_block_dim, "Can't create blas function with two StaticBlockDim expressions");
            static_assert(has_one_streaming, "Can't create blas function with two StaticBlockDim expressions");
            static_assert(has_one_streaming,
                          "Can't create blas function with more than one EnableInputStreaming expression");
            static_assert(has_one_with_pipeline,
                          "Can't create blas function with more than one WithPipeline expression");
            static_assert(has_one_required_mantissa_bits,
                          "Can't create blas function with more than one RequiredMantissaBits<> expression");

            // Operators checks

            // Side<> and Diag<> can only be used with TRSM.
            static constexpr bool valid_mm_description_no_trsm_ops =
                !has_function || (this_blas_function_v == function::TRSM) || !(has_side || has_diag || has_fill_mode);
            static_assert(valid_mm_description_no_trsm_ops,
                          "Operators Side<>, Diag<>, and FillMode<> can only be used with Function<function::TRSM>");

            // Arrangement and TransposeMode
            static_assert(!(has_arrangement && has_transpose_mode),
                          "Can't create blas function with Arrangement<> and TransposeMode<> expressions");

            // Leading dimensions check
            // NN --> >=LD(M, K, M)
            // TN --> >=LD(K, K, M)
            // NT --> >=LD(M, N, M)
            // TT --> >=LD(K, N, M)
            static constexpr bool valid_lda =
                !has_ld || !has_size || !has_function || !(this_blas_function_v == function::MM) || (this_blas_lda >= default_lda);
            static_assert(valid_lda || (this_blas_arrangement_a != arrangement::col_major),
                          "Incorrect leading dimension for A matrix, LDA must be greater than M");
            static_assert(valid_lda || (this_blas_arrangement_a == arrangement::col_major),
                          "Incorrect leading dimension for A matrix, LDA must be greater than K");
            static constexpr bool valid_ldb =
                !has_ld || !has_size || !has_function || !(this_blas_function_v == function::MM) || (this_blas_ldb >= default_ldb);
            static_assert(valid_ldb || (this_blas_arrangement_b != arrangement::col_major),
                          "Incorrect leading dimension for B matrix, LDB must be greater than K");
            static_assert(valid_ldb || (this_blas_arrangement_b == arrangement::col_major),
                          "Incorrect leading dimension for B matrix, LDB must be greater than N");
            static constexpr bool valid_ldc =
                !has_ld || !has_size || !has_function || !(this_blas_function_v == function::MM) || (this_blas_ldc >= default_ldc);
            static_assert(valid_ldc || (this_blas_arrangement_c != arrangement::col_major),
                          "Incorrect leading dimension for C matrix, LDC must be greater than M");
            static_assert(valid_ldc || (this_blas_arrangement_c == arrangement::col_major),
                          "Incorrect leading dimension for C matrix, LDC must be greater than N");
            // TRSM LD must be >= natural LD when specified.
            static constexpr bool valid_trsm_lda =
                !has_ld || !has_size || !has_function || !(this_blas_function_v == function::TRSM) ||
                (this_blas_trsm_lda >= this_blas_trsm_natural_lda);
            static_assert(valid_trsm_lda,
                          "Incorrect leading dimension for TRSM A matrix: LDA must be >= dim_a "
                          "(M for side::left, N for side::right)");
            static constexpr bool valid_trsm_ldb =
                !has_ld || !has_size || !has_function || !(this_blas_function_v == function::TRSM) ||
                (this_blas_trsm_ldb >= this_blas_trsm_natural_ldb);
            static_assert(valid_trsm_ldb,
                          "Incorrect leading dimension for TRSM B matrix: LDB must be >= M (col_major) or N (row_major)");

            // Size, precision, type, sm check
            static constexpr bool dont_check_if_size_fits_in_shared = true;

            // GEMM
            // Size
            static constexpr bool valid_size_for_block_gemm =
                dont_check_if_size_fits_in_shared || !has_size || !has_function || !has_sm ||
                !has_function || !(this_blas_function_v == function::MM) ||
                is_supported_logical_size_rmem_restrict<this_blas_precision,
                                                        this_blas_alignment,
                                                        this_blas_type_v,
                                                        this_blas_size,
                                                        this_blas_sm_v>::value;
            static_assert(valid_size_for_block_gemm,
                          "Provided size (M, N, K) for GEMM exceeds maximum supported for selected precision and type. "
                          "Matrices A, B, and C must fit into shared memory.");
            // LD
            static constexpr bool valid_ld_for_block_gemm = dont_check_if_size_fits_in_shared || !has_size || !has_ld ||
                                                            !has_function || !has_sm ||
                                                            !(this_blas_function_v == function::MM) ||
                                                            (is_supported_real_size<this_blas_sm_v,
                                                                                    this_blas_precision,
                                                                                    this_blas_alignment,
                                                                                    this_blas_type_v,
                                                                                    this_blas_a_size,
                                                                                    this_blas_b_size>::value);
            static_assert(
                valid_ld_for_block_gemm,
                "Provided leading dimensions for GEMM exceeds maximum supported for selected precision and type. "
                "Matrices A, B, and C must fit into shared memory.");

            using this_blas_block_dim = get_or_default_t<operator_type::block_dim, description_type, BlockDim<128>>;
            static constexpr auto this_blas_block_dim_v = this_blas_block_dim::value;
            
            static constexpr bool valid_block_dim =
                !has_block_dim || 
                (this_blas_block_dim::flat_size >= 32 && this_blas_block_dim::flat_size <= 1024);

            static_assert(valid_block_dim,
                          "Provided block dimension is invalid, BlockDim<> must have at least 32 threads, and can't "
                          "have more than 1024 threads.");

            // Precision
            static constexpr bool is_only_integral = (commondx::is_integral_v<typename this_blas_precision::a_type> and
                                                      commondx::is_integral_v<typename this_blas_precision::b_type> and
                                                      commondx::is_integral_v<typename this_blas_precision::c_type>);

            static constexpr bool is_only_floating_point =
                (commondx::is_floating_point_v<typename this_blas_precision::a_type> and
                 commondx::is_floating_point_v<typename this_blas_precision::b_type> and
                 commondx::is_floating_point_v<typename this_blas_precision::c_type>);

            static constexpr bool is_precision_coherent = is_only_integral or is_only_floating_point;

            static_assert(
                is_precision_coherent,
                "Precision operator cannot mix integral and floating point types, this effect can be achieved "
                "only by decoupling input and compute precisions");

            static constexpr bool is_complex_signed =
                this_blas_type_v == type::real or is_only_floating_point or
                commondx::is_signed_integral_v<typename this_blas_precision::a_type> and
                    commondx::is_signed_integral_v<typename this_blas_precision::b_type>;

            static_assert(is_complex_signed, "Complex BLAS type cannot be used with unsigned integral precisions");

            static constexpr bool is_accumulator_wide_enough =
                is_only_floating_point or
                // The accumulator is expected to be at least 2 orders of magnitude bigger
                // e.g. 8-8-32bit or 16-16-64bit
                (sizeof(typename this_blas_precision::a_type) * 4 <= sizeof(typename this_blas_precision::c_type) and
                 sizeof(typename this_blas_precision::b_type) * 4 <= sizeof(typename this_blas_precision::c_type));

            static_assert(is_accumulator_wide_enough,
                          "If integral computation is used, the accumulator type must be at least 4 times wider than "
                          "the input types, e.g. (int8_t, int8_t, int32_t)");

            // If either A or B are signed, then C must be signed
            static constexpr bool is_accumulator_correctly_signed =
                is_only_floating_point or ((commondx::is_unsigned_integral_v<typename this_blas_precision::a_type> and
                                            commondx::is_unsigned_integral_v<typename this_blas_precision::b_type>) or
                                           commondx::is_signed_integral_v<typename this_blas_precision::c_type>);

            static_assert(
                is_accumulator_correctly_signed,
                "If either A or B matrix are of signed integral type, then the C accumulator matrix must also "
                "be of signed integral type");


            static constexpr bool is_accumulator_precision_allowed = 
                cute::is_same_v<typename this_blas_precision::c_type, int32_t> or
                cute::is_same_v<typename this_blas_precision::c_type, uint32_t> or
                cute::is_same_v<typename this_blas_precision::c_type, int64_t> or
                cute::is_same_v<typename this_blas_precision::c_type, uint64_t> or
                cute::is_same_v<typename this_blas_precision::c_type, __half> or
                cute::is_same_v<typename this_blas_precision::c_type, float> or
                cute::is_same_v<typename this_blas_precision::c_type, double>;

            static_assert(is_accumulator_precision_allowed, "Accumulator precision must be one of the following: int32_t, uint32_t, int64_t, uint64_t, __half, float, double, please consult docs for more information");

            // TRSM specific constraints
            static constexpr bool block_dim_implies_block = !(has_thread && has_block_dim);
            static_assert(
                block_dim_implies_block,
                "BlockDim() cannot be used with Thread() - Thread()-level TRSM has no thread-block concept");

            static constexpr bool batches_per_block_implies_block = !(has_thread && has_batches_per_block);
            static_assert(
                batches_per_block_implies_block,
                "BatchesPerBlock() cannot be used with Thread() - Thread() always processes one batch per thread");
            
            static constexpr bool batches_per_block_implies_trsm = !has_batches_per_block || !has_function || (this_blas_function_v == function::TRSM);
            static_assert(batches_per_block_implies_trsm,
                "BatchesPerBlock() can only be used with Function<TRSM>");
            
            static constexpr bool has_correct_execution = not(has_block and has_thread);
            static_assert(has_correct_execution,
                "Can't use both Block() and Thread() operators in the same BLAS descriptor");

            static constexpr bool thread_implies_trsm = !has_thread || !has_function || (this_blas_function_v == function::TRSM);
            static_assert(thread_implies_trsm,
                "Thread() execution is only supported with Function<TRSM>();"
                " use Block() for GEMM and other functions");

            static constexpr bool side_implies_trsm = !has_side || !has_function || (this_blas_function_v == function::TRSM);
            static_assert(side_implies_trsm,
                "Side() can only be used with Function<TRSM>");

            static constexpr bool diag_implies_trsm = !has_diag || !has_function || (this_blas_function_v == function::TRSM);
            static_assert(diag_implies_trsm,
                "Diag() can only be used with Function<TRSM>");

            static constexpr bool fill_mode_implies_trsm = !has_fill_mode || !has_function || (this_blas_function_v == function::TRSM);
            static_assert(fill_mode_implies_trsm,
                "FillMode() can only be used with Function<TRSM>");


            static constexpr bool tile_implies_gemm = !has_tile || !has_function || (this_blas_function_v == function::MM);
            static_assert(tile_implies_gemm,
                "Tile() can only be used with Function<MM>");

            static constexpr bool static_block_dim_implies_gemm = !has_static_block_dim || !has_function || (this_blas_function_v == function::MM);
            static_assert(static_block_dim_implies_gemm,
                "StaticBlockDim() can only be used with Function<MM>");

            static constexpr bool streaming_implies_gemm = !has_streaming || !has_function || (this_blas_function_v == function::MM);
            static_assert(streaming_implies_gemm,
                "EnableInputStreaming() can only be used with Function<MM>");

            static constexpr bool with_pipeline_implies_gemm = !has_with_pipeline || !has_function || (this_blas_function_v == function::MM);
            static_assert(with_pipeline_implies_gemm,
                "WithPipeline() can only be used with Function<MM>");

            static constexpr bool required_mantissa_bits_implies_pipeline =
                !has_required_mantissa_bits || has_with_pipeline;
            static_assert(required_mantissa_bits_implies_pipeline,
                          "RequiredMantissaBits<X>() requires WithPipeline()");

            static constexpr bool required_mantissa_bits_implies_block =
                !has_required_mantissa_bits || has_block;
            static_assert(required_mantissa_bits_implies_block,
                          "RequiredMantissaBits<X>() requires Block()");

            static constexpr bool required_mantissa_bits_implies_gemm =
                !has_required_mantissa_bits || !has_function || (this_blas_function_v == function::MM);
            static_assert(required_mantissa_bits_implies_gemm,
                          "RequiredMantissaBits<X>() requires Function<MM>()");

            static constexpr bool required_mantissa_bits_implies_real =
                !has_required_mantissa_bits || (this_blas_type_v == type::real);
            static_assert(required_mantissa_bits_implies_real,
                          "RequiredMantissaBits<X>() supports only Type<type::real>()");

            static constexpr bool required_mantissa_bits_has_supported_precision =
                !has_required_mantissa_bits ||
                ((cute::is_same_v<typename this_blas_precision::a_type, float> or
                  cute::is_same_v<typename this_blas_precision::a_type, double>) and
                 (cute::is_same_v<typename this_blas_precision::a_type, typename this_blas_precision::b_type> and
                  cute::is_same_v<typename this_blas_precision::b_type, typename this_blas_precision::c_type>));
            static_assert(required_mantissa_bits_has_supported_precision,
                          "RequiredMantissaBits<X>() supports only Precision<float, float, float>() "
                          "or Precision<double, double, double>()");

            static constexpr bool transpose_mode_implies_gemm = !has_transpose_mode || !has_function || (this_blas_function_v == function::MM);
            static_assert(transpose_mode_implies_gemm,
                "TransposeMode() can only be used with Function<MM>");

            static constexpr bool trsm_has_acceptable_precision = 
                !has_function || (this_blas_function_v != function::TRSM) or
                ((cute::is_same_v<typename this_blas_precision::a_type, float> or 
                  cute::is_same_v<typename this_blas_precision::a_type, double>) and
                 (cute::is_same_v<typename this_blas_precision::a_type, typename this_blas_precision::b_type> and
                  cute::is_same_v<typename this_blas_precision::b_type, typename this_blas_precision::c_type>));
            static_assert(trsm_has_acceptable_precision,
                "TRSM requires either float or double and the same precision for A, B, and C");

            static constexpr bool trsm_size_only_has_2_fields = 
                !has_function || (this_blas_function_v != function::TRSM) or !has_size or this_blas_size_k_v == 1;

            static_assert(trsm_size_only_has_2_fields,
                "TRSM requires only M and N fields in Size<>");

            static constexpr bool trsm_ld_only_has_2_fields = 
                !has_function || (this_blas_function_v != function::TRSM) or !has_ld or this_blas_ldc == 1;
            static_assert(trsm_ld_only_has_2_fields,
                "TRSM requires only LDA and LDB fields in LeadingDimension<>");

            static constexpr bool trsm_precision_is_uniform = 
                !has_function || (this_blas_function_v != function::TRSM) or !has_precision or(cute::is_same_v<typename this_blas_precision::a_type, typename this_blas_precision::b_type> and 
                              cute::is_same_v<typename this_blas_precision::b_type, typename this_blas_precision::c_type>);
            static_assert(trsm_precision_is_uniform,
                "TRSM requires the same precision for A and B");


            static constexpr bool trsm_alignment_only_has_2_fields = !has_function || (this_blas_function_v != function::TRSM) or !has_alignment or (this_blas_alignment_c == 1);
            static_assert(trsm_alignment_only_has_2_fields,
                "TRSM requires only A and B fields in Alignment<>");

            // TRSM with accelerated SM targets (arch_specific / family_specific, e.g.
            // SM<900, arch_specific> for SM90a) requires CUDA Toolkit 13.2 or later.
#if defined(__CUDACC__)
            static constexpr bool trsm_accel_ctk_ok =
                !has_function || (this_blas_function_v != function::TRSM) ||
                (this_blas_sm_modifier_v == sm_modifier::generic) ||
                ((__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__) >= 1302);
            static_assert(trsm_accel_ctk_ok,
                "Due to NVCC bug (NVBUG5504462) TRSM with accelerated SM targets (e.g. SM<900, arch_specific> or "
                "SM<900, family_specific>) requires CUDA Toolkit 13.2 or later");
#endif

#ifdef CUBLASDX_NO_FATBIN_AVAILABLE
            static constexpr bool is_only_header_functionality_used = !has_function || (this_blas_function_v == function::MM);
            static_assert(is_only_header_functionality_used,
                "Only header functionality (GEMM) is supported when linking without the LTO fatbin. Please consult documentation "
                "for linking with TRSM (check cublasdx::cublasdx_fatbin and cublasdx::cublasdx_no_lto targets for more fine grained control) or "
                "use CUDA Toolkit 13.2 or later to enable TRSM on regular cublasdx::cublasdx target");
#endif
            /// ---- End of Constraints
        };

        template<>
        class blas_description<>: public commondx::detail::description_expression
        {
        };
    } // namespace detail
} // namespace cublasdx

#undef STRINGIFY
#undef XSTRINGIFY

#endif // CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP
