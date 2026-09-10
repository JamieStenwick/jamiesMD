// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_DATABASE_WORKAROUNDS_HPP
#define CUBLASDX_DATABASE_DATABASE_WORKAROUNDS_HPP

#include "cublasdx/database/gemm/mma_atoms.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {

            template<class MMAType>
            constexpr bool is_vectorized_fma() {
                return cute::is_same_v<MMAType, cublasdx::detail::SM100_2x1x1_F32F32F32F32> or
                       cute::is_same_v<MMAType, cublasdx::detail::SM100_1x2x1_F32F32F32F32>;
            }

            constexpr bool is_vectorized_fma(mma_atom atom) {
                return atom == mma_atom::SM100_2x1x1_F32F32F32F32 or atom == mma_atom::SM100_1x2x1_F32F32F32F32;
            }

            template<mma_atom Atom>
            struct convert_vector_fma_to_row_major {
                static_assert(is_vectorized_fma(Atom), "Atom is not vectorized");
                static_assert(Atom == mma_atom::SM100_2x1x1_F32F32F32F32,
                              "Currently only supports SM100_2x1x1_F32F32F32F32");
            };

            template<>
            struct convert_vector_fma_to_row_major<mma_atom::SM100_2x1x1_F32F32F32F32> {
                static constexpr mma_atom value = mma_atom::SM100_1x2x1_F32F32F32F32;
            };

            constexpr mma_atom apply_mma_workarounds(mma_atom                   atom,
                                                     [[maybe_unused]] const int                  m,
                                                     [[maybe_unused]] const int                  n,
                                                     [[maybe_unused]] const int                  k,
                                                     [[maybe_unused]] const int                  blockdim,
                                                     [[maybe_unused]] const int sm) {
                if (is_vectorized_fma(atom) and (sm != 1000 and sm != 1030)) {
                    return mma_atom::universal_fma;
                }

                return atom;
            }

        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_DATABASE_WORKAROUNDS_HPP
