// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_CHECK_HPP
#define CUBLASDX_DETAIL_BLAS_CHECK_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/device_info.hpp"

#include "cublasdx/operators.hpp"
#include "cublasdx/traits.hpp"
#include "cublasdx/detail/numerical.hpp"

#include "cublasdx/detail/pipeline/accumulator_mode.hpp"

namespace cublasdx {
    namespace detail {

        template<class LD, class Size, arrangement AArr, arrangement BArr, arrangement CArr>
        struct get_ld: public LD {
        };

        template<class Size, arrangement AArr, arrangement BArr, arrangement CArr>
        struct get_ld<void, Size, AArr, BArr, CArr> {
            static constexpr unsigned a = (AArr == col_major ? Size::m : Size::k);
            static constexpr unsigned b = (BArr == col_major ? Size::k : Size::n);
            static constexpr unsigned c = (CArr == col_major ? Size::m : Size::n);
        };

        template<unsigned int Arch,
                 class PA,
                 class PB,
                 class PC,
                 int            AAlignment,
                 int            BAlignment,
                 int            CAlignment,
                 cublasdx::type Type>
        constexpr bool is_supported_helper(unsigned a_size, unsigned b_size, unsigned c_size = 0) {
            unsigned type_multiplier        = (Type == cublasdx::type::real) ? 1 : 2;
            size_t   required_shared_memory = round_up(sizeof(PA) * type_multiplier * (a_size), BAlignment) +
                                              round_up(sizeof(PB) * type_multiplier * (b_size), CAlignment) +
                                              sizeof(PC) * type_multiplier * (c_size);
            // Cut of biggest cases to allow for pipeline to fit into shared
            constexpr int pipeline_safety_buffer = 256;
            return required_shared_memory + sizeof(pipeline_stage_scratch_t) <=
                   (commondx::device_info<Arch>::shared_memory() - pipeline_safety_buffer);
        }

        template<class Precision, class Alignment, cublasdx::type Type, class Size, unsigned int Arch>
        struct is_supported_logical_size_rmem_restrict:
            public COMMONDX_STL_NAMESPACE::bool_constant<is_supported_helper<Arch,
                                                                             typename Precision::a_type,
                                                                             typename Precision::b_type,
                                                                             typename Precision::c_type,
                                                                             Alignment::a,
                                                                             Alignment::b,
                                                                             Alignment::c,
                                                                             Type>(Size::m* Size::k,
                                                                                   Size::k* Size::n)> {
        };

        template<class Precision, class Alignment, cublasdx::type Type, class Size, unsigned int Arch>
        struct is_supported_logical_size_smem_restrict:
            public COMMONDX_STL_NAMESPACE::bool_constant<
                is_supported_helper<Arch,
                                    typename Precision::a_type,
                                    typename Precision::b_type,
                                    typename Precision::c_type,
                                    Alignment::a,
                                    Alignment::b,
                                    Alignment::c,
                                    Type>(Size::m* Size::k, Size::k* Size::n, Size::m* Size::n)> {
        };

        // Checks if matrices A, B, C fits into shared memory
        template<unsigned int Arch,
                 class Precision,
                 class Alignment,
                 cublasdx::type Type,
                 unsigned int   ASize,
                 unsigned int   BSize,
                 unsigned int   CSize = 0>
        struct is_supported_real_size:
            public COMMONDX_STL_NAMESPACE::bool_constant<is_supported_helper<Arch,
                                                                             typename Precision::a_type,
                                                                             typename Precision::b_type,
                                                                             typename Precision::c_type,
                                                                             Alignment::a,
                                                                             Alignment::b,
                                                                             Alignment::c,
                                                                             Type>(ASize, BSize, CSize)> {
        };

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
        struct is_supported_restrict_impl {
            using ld_getter           = get_ld<LD, Size, AArr, BArr, CArr>;
            static constexpr auto lda = ld_getter::a;
            static constexpr auto ldb = ld_getter::b;
            static constexpr auto ldc = ld_getter::c;

            static constexpr unsigned int m = Size::m;
            static constexpr unsigned int n = Size::n;
            static constexpr unsigned int k = Size::k;

            static constexpr auto a_size = lda * ((AArr == arrangement::col_major) ? k : m);
            static constexpr auto b_size = ldb * ((BArr == arrangement::col_major) ? n : k);
            static constexpr auto c_size = ldc * ((CArr == arrangement::col_major) ? n : m);

            static constexpr auto a_alignment = Alignment::a;
            static constexpr auto b_alignment = Alignment::b;
            static constexpr auto c_alignment = Alignment::c;

            // Static LD methods
            static constexpr bool rmem_value() {
                return is_supported_helper<Arch, PA, PB, PC, a_alignment, b_alignment, c_alignment, Type>(a_size,
                                                                                                          b_size);
            }

            static constexpr bool smem_value() {
                return is_supported_helper<Arch, PA, PB, PC, a_alignment, b_alignment, c_alignment, Type>(
                    a_size, b_size, c_size);
            }

            // Dynamic LD methods
            static constexpr bool rmem_value(unsigned dyn_lda, unsigned dyn_ldb) {
                const auto dyn_a_size = dyn_lda * ((AArr == arrangement::col_major) ? k : m);
                const auto dyn_b_size = dyn_ldb * ((BArr == arrangement::col_major) ? n : k);
                return is_supported_helper<Arch, PA, PB, PC, a_alignment, b_alignment, c_alignment, Type>(dyn_a_size,
                                                                                                          dyn_b_size);
            }

            static constexpr bool smem_value(unsigned dyn_lda, unsigned dyn_ldb, unsigned dyn_ldc) {
                const auto dyn_a_size = dyn_lda * ((AArr == arrangement::col_major) ? k : m);
                const auto dyn_b_size = dyn_ldb * ((BArr == arrangement::col_major) ? n : k);
                const auto dyn_c_size = dyn_ldc * ((CArr == arrangement::col_major) ? n : m);
                return is_supported_helper<Arch, PA, PB, PC, a_alignment, b_alignment, c_alignment, Type>(
                    dyn_a_size, dyn_b_size, dyn_c_size);
            }
        };
    } // namespace detail

    // Check if a description is supported on a given CUDA architecture
    template<class Description,
             unsigned int Architecture,
             class APrecision = precision_of_a_t<Description>,
             class BPrecision = precision_of_b_t<Description>,
             class CPrecision = precision_of_c_t<Description>>
    struct is_supported_rmem_restrict {
        using type = detail::is_supported_restrict_impl<
            APrecision,
            BPrecision,
            CPrecision,
            detail::get_or_default_t<operator_type::alignment,
                                     Description,
                                     Alignment<sizeof(APrecision), sizeof(BPrecision), sizeof(CPrecision)>>,
            type_of_v<Description>,
            arrangement_of<Description>::a,
            arrangement_of<Description>::b,
            arrangement_of<Description>::c,
            detail::get_t<operator_type::size, Description>,
            detail::get_t<operator_type::ld, Description>,
            Architecture>;
        static constexpr bool value = type::rmem_value();

        static bool dynamic_value(unsigned lda, unsigned ldb) { return type::rmem_value(lda, ldb); }
    };

    template<class Description,
             unsigned int Architecture,
             class APrecision = precision_of_a_t<Description>,
             class BPrecision = precision_of_b_t<Description>,
             class CPrecision = precision_of_c_t<Description>>
    struct is_supported_smem_restrict {
        using type = detail::is_supported_restrict_impl<
            APrecision,
            BPrecision,
            CPrecision,
            detail::get_or_default_t<operator_type::alignment,
                                     Description,
                                     Alignment<sizeof(APrecision), sizeof(BPrecision), sizeof(CPrecision)>>,
            type_of_v<Description>,
            arrangement_of<Description>::a,
            arrangement_of<Description>::b,
            arrangement_of<Description>::c,
            detail::get_t<operator_type::size, Description>,
            detail::get_t<operator_type::ld, Description>,
            Architecture>;
        static constexpr bool value = type::smem_value();

        static bool dynamic_value(unsigned lda, unsigned ldb, unsigned ldc) { return type::smem_value(lda, ldb, ldc); }
    };


    template<class Description, unsigned int Architecture, class... Precisions>
    inline constexpr bool is_supported_rmem_restrict_v =
        is_supported_rmem_restrict<Description, Architecture, Precisions...>::value;

    template<class Description, unsigned int Architecture, class... Precisions>
    inline constexpr bool is_supported_smem_restrict_v =
        is_supported_smem_restrict<Description, Architecture, Precisions...>::value;
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_BLAS_CHECK_HPP
