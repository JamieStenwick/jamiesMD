// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_COMPLEX_MMA_TRAITS_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_COMPLEX_MMA_TRAITS_HPP

#include <cute/config.hpp>

#include "cublasdx/detail/system_checks.hpp"

#include "commondx/traits/numeric_traits.hpp"

#include "cublasdx/detail/cute/extension/arch/complex_mma.hpp"

namespace cute {
    template<class RealMMA>
    struct MMA_Traits<cublasdx::detail::ComplexMMA<RealMMA>>: MMA_Traits<RealMMA> {
        using ValTypeD = complex<typename MMA_Traits<RealMMA>::ValTypeD>;
        using ValTypeA = complex<typename MMA_Traits<RealMMA>::ValTypeA>;
        using ValTypeB = complex<typename MMA_Traits<RealMMA>::ValTypeB>;
        using ValTypeC = complex<typename MMA_Traits<RealMMA>::ValTypeC>;

        template<class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
        CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const&          traits,
                                                          Tensor<TD, DLayout>&       D,
                                                          Tensor<TA, ALayout> const& A,
                                                          Tensor<TB, BLayout> const& B,
                                                          Tensor<TC, CLayout> const& C) {
            static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
            static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
            static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
            static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

            using complex_mma = cublasdx::detail::ComplexMMA<RealMMA>;

            // Register value types from the MMA_Operation register arrays
            using RegTypeD = typename remove_extent<typename complex_mma::DRegisters>::type;
            using RegTypeA = typename remove_extent<typename complex_mma::ARegisters>::type;
            using RegTypeB = typename remove_extent<typename complex_mma::BRegisters>::type;
            using RegTypeC = typename remove_extent<typename complex_mma::CRegisters>::type;

            constexpr int RegNumD = extent<typename complex_mma::DRegisters>::value;
            constexpr int RegNumA = extent<typename complex_mma::ARegisters>::value;
            constexpr int RegNumB = extent<typename complex_mma::BRegisters>::value;
            constexpr int RegNumC = extent<typename complex_mma::CRegisters>::value;

            Tensor rA = recast<RegTypeA>(A);
            Tensor rB = recast<RegTypeB>(B);
            Tensor rD = recast<RegTypeD>(D);
            Tensor rC = recast<RegTypeC>(C);

            CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA> {});
            CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB> {});
            CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD> {});
            CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC> {});

            complex_mma::fma(rD, rA, rB, rC);
        }
    };
} // namespace cute

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ATOM_COMPLEX_MMA_TRAITS_HPP
