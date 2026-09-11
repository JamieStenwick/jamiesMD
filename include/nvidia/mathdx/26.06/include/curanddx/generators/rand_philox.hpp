// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DATABASE_RAND_PHILOX_HPP
#define CURANDDX_DATABASE_RAND_PHILOX_HPP

#include "curanddx/detail/common.hpp"

namespace curanddx {
    namespace detail {

        inline constexpr unsigned int philox4x32_w32_0   = 0x9E3779B9U;
        inline constexpr unsigned int philox4x32_w32_1   = 0xBB67AE85U;
        inline constexpr unsigned int philox4x32_m4x32_0 = 0xD2511F53U;
        inline constexpr unsigned int philox4x32_m4x32_1 = 0xCD9E8D57U;

        CURANDDX_QUALIFIERS unsigned int mulhilo32(unsigned int a, unsigned int b, unsigned int* hip) {
            *hip = __umulhi(a, b);
            return a * b;
        }

        CURANDDX_QUALIFIERS uint4 single_round(uint4 ctr, uint2 key) {
            unsigned int hi0;
            unsigned int hi1;
            unsigned int lo0 = mulhilo32(philox4x32_m4x32_0, ctr.x, &hi0);
            unsigned int lo1 = mulhilo32(philox4x32_m4x32_1, ctr.z, &hi1);

            uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
            return ret;
        }

        template<unsigned int Rounds>
        CURANDDX_QUALIFIERS uint4 multiple_rounds(uint4 c, uint2 k) {
            for (unsigned int i = 0; i < Rounds - 1; i++) {
                c = single_round(c, k); // 1
                k.x += philox4x32_w32_0;
                k.y += philox4x32_w32_1;
            }
            return single_round(c, k); // Rounds
        }

        template<unsigned int Rounds>
        struct philox4x32_native_state {
            static constexpr unsigned int rounds = Rounds;

            uint4 ctr;
            uint2 key;

            CURANDDX_QUALIFIERS void philox_state_incr() {
                if (++ctr.x)
                    return;
                if (++ctr.y)
                    return;
                if (++ctr.z)
                    return;
                ++ctr.w;
            }

            CURANDDX_QUALIFIERS void philox_state_incr(unsigned long long n) {
                unsigned int nlo = (unsigned int)(n);
                unsigned int nhi = (unsigned int)(n >> 32);

                ctr.x += nlo;
                if (ctr.x < nlo)
                    nhi++;

                ctr.y += nhi;
                if (nhi <= ctr.y)
                    return;
                if (++ctr.z)
                    return;
                ++ctr.w;
            }

            CURANDDX_QUALIFIERS void philox_state_incr_hi(unsigned long long n) {
                unsigned int nlo = (unsigned int)(n);
                unsigned int nhi = (unsigned int)(n >> 32);

                ctr.z += nlo;
                if (ctr.z < nlo)
                    nhi++;

                ctr.w += nhi;
            }

            // offset is the total # of 128bits generated with a single generate4() call
            CURANDDX_QUALIFIERS void skip_offset(unsigned long long n) { philox_state_incr(n); }

            CURANDDX_QUALIFIERS void skip_subsequence(unsigned long long n) { philox_state_incr_hi(n); }

            CURANDDX_QUALIFIERS void init(unsigned long long seed,
                                          unsigned long long subsequence,
                                          unsigned long long offset) {
                ctr   = make_uint4(0, 0, 0, 0);
                key.x = (unsigned int)seed;
                key.y = (unsigned int)(seed >> 32);

                skip_subsequence(subsequence);
                skip_offset(offset);
            }

            CURANDDX_QUALIFIERS uint4 generate4() {
                auto tmp = multiple_rounds<Rounds>(ctr, key);
                philox_state_incr();
                return tmp;
            }
        };
    } // namespace detail
} // namespace curanddx


#endif // CURANDDX_DATABASE_RAND_PHILOX_HPP
