// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DATABASE_RAND_PCG_HPP
#define CURANDDX_DATABASE_RAND_PCG_HPP

#include <commondx/detail/stl/cstdint.hpp>
#include "curanddx/detail/common.hpp"

namespace curanddx {
    namespace detail {

        struct pcg_state {
        private:
            static constexpr unsigned long long multiple_constant = 6364136223846793005ULL;
            static constexpr unsigned long long default_seed = 0x853c49e6748fea9bULL;
            static constexpr unsigned long long default_increment = 0xda3e39cb94b95bdbULL;

            unsigned long long m_internal_state {default_seed};
            unsigned long long m_stream {0ULL};
            unsigned long long m_increment {default_increment};

        public:
            CURANDDX_QUALIFIERS void skip_offset(unsigned long long offset) {
                if (offset > 0) {
                    unsigned long long acc_mult = 1;
                    unsigned long long acc_plus = 0;
                    unsigned long long cur_mult = multiple_constant;
                    unsigned long long cur_plus = m_stream;
                    while (offset > 0) {
                        if (offset & 1) {
                            acc_mult *= cur_mult;
                            acc_plus = acc_plus * cur_mult + cur_plus;
                        }
                        cur_plus = (cur_mult + 1) * cur_plus;
                        cur_mult *= cur_mult;
                        offset >>= 1;
                    }

                    m_internal_state = acc_mult * m_internal_state + acc_plus;
                }
            }

            CURANDDX_QUALIFIERS void skip_subsequence(unsigned long long n) { m_stream = n * 2 + m_increment; }
            CURANDDX_QUALIFIERS void set_increment(unsigned long long n) { m_increment = n; }

            CURANDDX_QUALIFIERS void init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset) {
                m_internal_state = seed;

                skip_subsequence(subsequence);
                skip_offset(offset);
            }

            CURANDDX_QUALIFIERS unsigned int randombits32() {
                uint64_t oldstate = m_internal_state;

                // Advance internal_state
                m_internal_state = oldstate * multiple_constant + m_stream;

                // Calculate output function (XSH RR), uses old internal_state for max ILP
                uint32_t xorshifted = uint32_t(((oldstate >> 18U) ^ oldstate) >> 27U);
                uint32_t rot        = uint32_t(oldstate >> 59U);

                uint32_t ret = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

                return ret;
            }

            CURANDDX_QUALIFIERS unsigned long long randombits64() {
                uint32_t ret_hi = randombits32();
                uint32_t ret_lo = randombits32();

                return (uint64_t(ret_hi) << 32U) | uint64_t(ret_lo);
            }
        };

    } // namespace detail
} // namespace curanddx


#endif // CURANDDX_DATABASE_RAND_PCG_HPP
