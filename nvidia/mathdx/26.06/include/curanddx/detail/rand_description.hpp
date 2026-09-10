// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_RAND_DESCRIPTION_HPP
#define CURANDDX_DETAIL_RAND_DESCRIPTION_HPP

#include "commondx/traits/detail/get.hpp"
#include "commondx/detail/expressions.hpp"

#include "curanddx/operators.hpp"
#include "curanddx/generators/select_generator_type.hpp"
#include "curanddx/traits/detail/description_traits.hpp"

namespace curanddx {
    namespace detail {

        template<class... Operators>
        class rand_operator_wrapper: public commondx::detail::description_expression
        {};

        template<class... Operators>
        class rand_description: public commondx::detail::description_expression
        {
            using description_type = rand_operator_wrapper<Operators...>;

        protected:
            // True if description is complete description
            static constexpr bool is_complete = is_complete_description<description_type>::value;

            // Generator
            // * Default value: pcg
            using this_rand_generator =
                get_or_default_t<operator_type::generator, description_type, default_curanddx_generator_operator>;
            static constexpr auto this_rand_generator_v = this_rand_generator::value;

            // Philox round
            // * Default = 10
            using this_rand_philox_rounds                   = get_or_default_t<operator_type::philox_rounds,
                                                             description_type,
                                                             default_curanddx_philox_rounds_operator>;
            static constexpr auto this_rand_philox_rounds_v = this_rand_philox_rounds::value;

            using this_rand_state_type =
                typename select_generator_type<this_rand_generator_v, this_rand_philox_rounds_v>::state_type;
            using this_rand_result_type =
                typename select_generator_type<this_rand_generator_v, this_rand_philox_rounds_v>::result_type;
            using this_rand_offset_type =
                typename select_generator_type<this_rand_generator_v, this_rand_philox_rounds_v>::offset_type;

            // For SOBOL
            using this_sobol_direction_vector_type =
                typename select_generator_type<this_rand_generator_v>::direction_vector_type;
            using this_sobol_scrambled_const_type =
                typename select_generator_type<this_rand_generator_v>::scrambled_const_type;

            // Ordering
            // * Default value: strict
            using this_rand_ordering =
                get_or_default_t<operator_type::ordering, description_type, default_curanddx_ordering_operator>;
            static constexpr auto this_rand_ordering_v = this_rand_ordering::value;

            // SM, no default
            static constexpr bool has_sm = has_operator<operator_type::sm, description_type>::value;

            /// ---- Constraints
            // We can only have one of each option
            static constexpr bool has_one_generator =
                has_at_most_one_of<operator_type::generator, description_type>::value;
            static constexpr bool has_one_sm = has_at_most_one_of<operator_type::sm, description_type>::value;
            static constexpr bool has_one_block_dim =
                has_at_most_one_of<operator_type::block_dim, description_type>::value;
            static constexpr bool has_one_grid_dim =
                has_at_most_one_of<operator_type::grid_dim, description_type>::value;

            static_assert(has_one_generator, "Can't create rand generator with two Generator<> expressions");
            static_assert(has_one_sm, "Can't create rand sm with two SM<> expressions");
            static_assert(has_one_block_dim, "Can't create rand block dim with two BlockDim<> expressions");
            static_assert(has_one_grid_dim, "Can't create rand grid dim with two GridDim<> expressions");


            static constexpr bool is_generator_pseudo_random = is_pseudo_random<this_rand_generator_v>::value;
            static constexpr bool is_generator_sobol         = is_sobol<this_rand_generator_v>::value;
            static constexpr bool is_generator_32bits        = is_32bits<this_rand_generator_v>::value;
            static constexpr bool is_generator_64bits        = is_64bits<this_rand_generator_v>::value;
            static constexpr bool is_generator_philox        = is_philox<this_rand_generator_v>::value;

            // Philox Rounds can only be used with Philox generator
            static constexpr bool has_philox_rounds =
                has_operator<operator_type::philox_rounds, description_type>::value;
            static constexpr bool valid_philox_rounds = !(has_philox_rounds && !is_generator_philox);
            static_assert(valid_philox_rounds, "PhiloxRounds operator can only be used with Philox generator");


            /// ---- End of Constraints
        };

        template<>
        class rand_description<>: public commondx::detail::description_expression
        {};
    } // namespace detail
} // namespace curanddx


#endif // CURANDDX_DETAIL_RAND_DESCRIPTION_HPP
