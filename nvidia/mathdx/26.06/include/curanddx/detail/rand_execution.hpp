// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_RAND_EXECUTION_HPP
#define CURANDDX_DETAIL_RAND_EXECUTION_HPP

#include "curanddx/generators/rand_generator.hpp"

namespace curanddx {
    namespace detail {

        //============================
        // rand_execution base
        //============================
        template<class... Operators>
        class rand_execution:
            public rand_description<Operators...>,
            public commondx::detail::execution_description_expression
        {
            using base_type = rand_description<Operators...>;
            using this_type = rand_execution<Operators...>;

        public:
            /// ---- Constraints
            // We need Thread, Block, or Grid operator to be specified exactly once
            static constexpr bool is_thread_execution = has_n_of<1, operator_type::thread, this_type>::value;
            static constexpr bool is_block_execution  = has_n_of<1, operator_type::block, this_type>::value;
            static constexpr bool is_device_execution = has_n_of<1, operator_type::device, this_type>::value;

            static constexpr bool is_generator_pseudo_random = base_type::is_generator_pseudo_random;
            static constexpr bool is_generator_sobol         = base_type::is_generator_sobol;
            static constexpr bool is_generator_32bits        = base_type::is_generator_32bits;
            static constexpr bool is_generator_64bits        = base_type::is_generator_64bits;
            static constexpr bool is_generator_philox        = base_type::is_generator_philox;
        };

        //============================
        // rand_thread_execution
        //============================
        template<class... Operators>
        class rand_thread_execution: public rand_execution<Operators...>
        {
            using base_type = rand_execution<Operators...>;
            using this_type = rand_thread_execution<Operators...>;
            using state_type            = typename base_type::this_rand_state_type;

            template<class RNG>
            friend class rand_generator;

            using rand_generator_type   = rand_generator<this_type>;
            
        public:
            using bitgenerator_result_type = typename base_type::this_rand_result_type;
            using offset_type              = typename base_type::this_rand_offset_type;

            using direction_vector_type = typename base_type::this_sobol_direction_vector_type;
            using scrambled_const_type  = typename base_type::this_sobol_scrambled_const_type;

            __device__ rand_thread_execution() {};

            // In case users allocate and initalized state_type* in a host function
            //CURANDDX_QUALIFIERS rand_thread_execution(state_type& state) { m_generator() = state; }

            CURANDDX_QUALIFIERS rand_generator_type&       get_generator() { return m_generator; }
            CURANDDX_QUALIFIERS const rand_generator_type& get_generator() const { return m_generator; }

            CURANDDX_QUALIFIERS rand_thread_execution(const unsigned long long seed,
                                             const unsigned long long subsequence,
                                             const offset_type offset) {
                init(seed, subsequence, offset);
            }

            //---------------------------
            // "int" Functions
            //---------------------------
            // Init functions for pseudo-random generators except MTGP, initialized with host API only
            template<class RNG = this_type>
            CURANDDX_QUALIFIERS auto init(const unsigned long long seed,
                                 const unsigned long long subsequence,
                                 const offset_type offset)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<(RNG::is_complete && (RNG::is_generator_pseudo_random &&
                                                                             generator_of_v<RNG> != mtgp32)),
                                                       void> {
                return m_generator.init(seed, subsequence, offset);
            }

            //---------------------------
            // "generate" Functions, only generates random bits
            //---------------------------
            template<class RNG = this_type>
            CURANDDX_QUALIFIERS auto generate() -> COMMONDX_STL_NAMESPACE::enable_if_t<RNG::is_complete && !RNG::is_generator_philox, bitgenerator_result_type> {
                return m_generator.generate();
            }

            template<class RNG = this_type>
            CURANDDX_QUALIFIERS auto generate4() -> COMMONDX_STL_NAMESPACE::enable_if_t<RNG::is_complete && RNG::is_generator_philox, bitgenerator_result_type> {
                return m_generator.generate4();
            }

            //---------------------------
            // "skip" Functions
            //---------------------------
            template<class RNG = this_type>
            CURANDDX_QUALIFIERS auto skip_offset(const offset_type n) -> COMMONDX_STL_NAMESPACE::enable_if_t<
                (RNG::is_complete &&
                 (generator_of_v<RNG> == xorwow || RNG::is_generator_philox || generator_of_v<RNG> == mrg32k3a ||
                  generator_of_v<RNG> == sobol32 || generator_of_v<RNG> == sobol64 || generator_of_v<RNG> == pcg)),
                void> {
                return m_generator.skip_offset(n);
            }
            template<class RNG = this_type>
            CURANDDX_QUALIFIERS auto skip_subsequence(const unsigned long long n) -> COMMONDX_STL_NAMESPACE::enable_if_t<
                (RNG::is_complete && (generator_of_v<RNG> == mrg32k3a || generator_of_v<RNG> == xorwow ||
                                      RNG::is_generator_philox || generator_of_v<RNG> == pcg)),
                void> {
                m_generator.skip_subsequence(n);
            }
            template<class RNG = this_type>
            CURANDDX_QUALIFIERS auto skip_sequence(const unsigned long long n)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<(RNG::is_complete && generator_of_v<RNG> == mrg32k3a), void> {
                m_generator.skip_sequence(n);
            }


        protected:
            rand_generator_type m_generator;
        };

        //+++++++++++++++++++++++++++++
        // rand_sobol_thread_execution
        //+++++++++++++++++++++++++++++
        template<class... Operators>
        class rand_sobol_thread_execution: public rand_thread_execution<Operators...>
        {
            using base_type = rand_thread_execution<Operators...>;
            using this_type = rand_sobol_thread_execution<Operators...>;

            static_assert(this_type::is_generator_sobol, "rand_sobol_thread_execution only applies for sobol");

        public:
            using direction_vector_type = typename base_type::this_sobol_direction_vector_type;
            using scrambled_const_type  = typename base_type::this_sobol_scrambled_const_type;
            using offset_type           = typename base_type::this_rand_offset_type;

            __device__ rand_sobol_thread_execution() {};

            CURANDDX_QUALIFIERS rand_sobol_thread_execution(const unsigned int          dim,
                                                   direction_vector_type*      direction_vectors,
                                                   const offset_type           offset,
                                                   const scrambled_const_type* scrambled_consts = nullptr) {
                init(dim, direction_vectors, offset, scrambled_consts);
            }

            CURANDDX_QUALIFIERS void init(const unsigned int          dim,
                                 direction_vector_type*      direction_vectors,
                                 const offset_type           offset,
                                 const scrambled_const_type* scrambled_consts = nullptr) {
                this->m_generator.init_sobol(dim, direction_vectors, offset, scrambled_consts);
            }
        };


        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool has_thread_operator =
                has_operator<operator_type::thread, rand_execution<Operators...>>::value;
            static constexpr bool has_block_operator =
                has_operator<operator_type::block, rand_execution<Operators...>>::value;
            static constexpr bool has_device_operator =
                has_operator<operator_type::device, rand_execution<Operators...>>::value;
            
            // disable block or device operator for 0.1.0
            static_assert(!has_block_operator && !has_device_operator, "cuRANDDx 0.1.0 only supports thread execution operator");

            static constexpr bool is_sobol = rand_execution<Operators...>::is_generator_sobol;

            static constexpr bool has_execution_operator =
                has_block_operator || has_thread_operator || has_device_operator;

            using rand_thread_execution_type       = rand_thread_execution<Operators...>;
            using rand_sobol_thread_execution_type = rand_sobol_thread_execution<Operators...>;

            using description_type = rand_description<Operators...>;

            using execution_type = COMMONDX_STL_NAMESPACE::
                conditional_t<is_sobol, rand_sobol_thread_execution_type, rand_thread_execution_type>;


        public:
            using type = typename COMMONDX_STL_NAMESPACE::
                conditional<has_execution_operator, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<
            commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
            detail::make_description_t<Operator1, Operator2>>::type {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::rand_description<Operators1...>&,
                                                       const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator2>::value,
                                                      detail::make_description_t<Operators1..., Operator2>>::type {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::rand_description<Operators2...>&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator1>::value,
                                                      detail::make_description_t<Operator1, Operators2...>>::type {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::rand_description<Operators1...>&,
                                                       const detail::rand_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace curanddx

#endif // CURANDDX_DETAIL_RAND_EXECUTION_HPP
