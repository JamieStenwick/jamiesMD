// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_RAND_GENERATOR_HPP
#define CURANDDX_DETAIL_RAND_GENERATOR_HPP

#include "curanddx/traits/rand_traits.hpp"

namespace curanddx {
    namespace detail {

        template<class RNG>
        class rand_generator
        {
            using state_type  = typename RNG::state_type;
            using offset_type = typename RNG::offset_type;

        public:
            CURANDDX_QUALIFIERS state_type&       operator()() { return m_state; }
            CURANDDX_QUALIFIERS const state_type& operator()() const { return m_state; }

            CURANDDX_QUALIFIERS void init(const unsigned long long seed,
                                          const unsigned long long subsequence,
                                          const offset_type        offset) {
                if constexpr (RNG::is_generator_philox || generator_of_v<RNG> == pcg) {
                    m_state.init(seed, subsequence, offset);
                } else {
                    ::curand_init(seed, subsequence, offset, &m_state);
                }
            }

            CURANDDX_QUALIFIERS auto init_sobol(const unsigned int                        dim,
                                                typename RNG::direction_vector_type*      direction_vectors,
                                                const offset_type                         offset,
                                                const typename RNG::scrambled_const_type* scrambled_consts = nullptr) {
                if constexpr (generator_of_v<RNG> == sobol32 || generator_of_v<RNG> == sobol64) {
                    ::curand_init(direction_vectors[dim], offset, &m_state);
                } else {
                    ::curand_init(direction_vectors[dim], scrambled_consts[dim], offset, &m_state);
                }
            }

            CURANDDX_QUALIFIERS auto generate() {
                static_assert(!RNG::is_generator_philox, "generate() is not supported for Philox generator");
                typename RNG::bitgenerator_result_type result;
                if constexpr (generator_of_v<RNG> == pcg) {
                    result = m_state.randombits32();
                } else {
                    result = ::curand(&m_state);
                }
                return result;
            }

            CURANDDX_QUALIFIERS auto generate4() {
                static_assert(RNG::is_generator_philox, "generate4() is only supported for Philox generator");
                uint4 result = m_state.generate4();
                return result;
            }

            //--------------- skip n offset. One offset is a single RNG::result_type from one call of generate()
            // Only support XORWOW, MRG, Philox, pcg, sobol32, and sobol64 generator
            CURANDDX_QUALIFIERS auto skip_offset(const offset_type n) {
                if constexpr (RNG::is_generator_philox || generator_of_v<RNG> == pcg) {
                    m_state.skip_offset(n);
                } else {
                    ::skipahead(n, &m_state);
                }
            }

            //--------------- skip n subsequence
            // in cuRand device API, MRG skipahead_sequence is really skipahead subsequence
            CURANDDX_QUALIFIERS auto skip_subsequence(const unsigned long long n) {
                if constexpr (generator_of_v<RNG> == mrg32k3a) {
                    ::skipahead_subsequence(n, &m_state);
                } else if constexpr (RNG::is_generator_philox || generator_of_v<RNG> == pcg) {
                    m_state.skip_subsequence(n);
                } else {
                    ::skipahead_sequence(n, &m_state);
                }
            }

            //--------------- skip n sequence
            // Only MRG generator
            CURANDDX_QUALIFIERS auto skip_sequence(const unsigned long long n) { ::skipahead_sequence(n, &m_state); }


        protected:
            state_type m_state;
        };


    } // namespace detail
} // namespace curanddx

#endif // CURANDDX_DETAIL_RAND_GENERATOR_HPP
