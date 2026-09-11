// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_RAND_TRAITS_HPP
#define CURANDDX_RAND_TRAITS_HPP

#include "curanddx/detail/rand_description_fd.hpp"
#include "curanddx/operators.hpp"
#include "curanddx/traits/detail/description_traits.hpp"

#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/dx_traits.hpp"

namespace commondx {
    // Partial specialization for cuRANDx as generic is_dx_execution_expression doesn't have support for operator_type::device/Device
    // since not all Dx libraries have device-wide execution.
    template<class Description>
    struct is_dx_execution_expression<curanddx::operator_type, Description> {
        static constexpr auto device        = detail::has_operator<curanddx::operator_type, curanddx::operator_type::device, Description>::value;
        static constexpr auto is_device_op  = COMMONDX_STL_NAMESPACE::is_same<Description, Device>::value;
        static constexpr auto block         = detail::has_operator<curanddx::operator_type, curanddx::operator_type::block, Description>::value;
        static constexpr auto is_block_op   = COMMONDX_STL_NAMESPACE::is_same<Description, Block>::value;
        static constexpr auto thread        = detail::has_operator<curanddx::operator_type, curanddx::operator_type::thread, Description>::value;
        static constexpr auto is_thread_op  = COMMONDX_STL_NAMESPACE::is_same<Description, Thread>::value;

    public:
        using value_type                  = bool;
        static constexpr value_type value = is_dx_expression<Description>::value && ((device && !is_device_op) || (block && !is_block_op) || (thread && !is_thread_op));
        constexpr operator value_type() const noexcept { return value; }
    };
} // namespace commondx

namespace curanddx {
    // generator_of
    template<class Description>
    using generator_of =
        detail::get_or_default_t<operator_type::generator, Description, detail::default_curanddx_generator_operator>;
    template<class Description>
    inline constexpr auto generator_of_v = generator_of<Description>::value;

    // philox rounds of
    template<class Description>
    using philox_rounds_of = detail::
        get_or_default_t<operator_type::philox_rounds, Description, detail::default_curanddx_philox_rounds_operator>;
    template<class Description>
    inline constexpr auto philox_rounds_of_v = philox_rounds_of<Description>::value;

    // // mode of
    // template<class Description>
    // using mode_of = detail::get_or_default_t<operator_type::mode, Description, detail::default_curanddx_mode_operator>;
    // template<class Description>
    // inline constexpr auto mode_of_v = mode_of<Description>::value;

    // ordering_of
    template<class Description>
    using ordering_of =
        detail::get_or_default_t<operator_type::ordering, Description, detail::default_curanddx_ordering_operator>;
    template<class Description>
    inline constexpr auto ordering_of_v = ordering_of<Description>::value;

    // sm_of
    template<class Description>
    using sm_of = commondx::sm_of<operator_type, Description>;
    template<class Description>
    inline constexpr unsigned int sm_of_v = sm_of<Description>::value;

    // block_dim_of
    template<class Description>
    using block_dim_of = commondx::block_dim_of<operator_type, Description>;
    template<class Description>
    inline constexpr dim3 block_dim_of_v = block_dim_of<Description>::value;

    // grid_dim_of
    template<class Description>
    using grid_dim_of = commondx::grid_dim_of<operator_type, Description>;
    template<class Description>
    inline constexpr dim3 grid_dim_of_v = grid_dim_of<Description>::value;

    // is_rand
    template<class Description>
    using is_rand = commondx::is_dx_expression<Description>;
    template<class Description>
    inline constexpr bool is_rand_v = commondx::is_dx_expression<Description>::value;

    // is_rand_execution
    template<class Description>
    using is_rand_execution = commondx::is_dx_execution_expression<operator_type, Description>;
    template<class Description>
    inline constexpr bool is_rand_execution_v = commondx::is_dx_execution_expression<operator_type, Description>::value;

    // is_complete_rand
    template<class Description>
    using is_complete_rand = commondx::is_complete_dx_expression<Description, detail::is_complete_description>;
    template<class Description>
    inline constexpr bool is_complete_rand_v =
        commondx::is_complete_dx_expression<Description, detail::is_complete_description>::value;

    // true if is_complete_rand is true && is_rand_execution is true
    template<class Description>
    using is_complete_rand_execution =
        commondx::is_complete_dx_execution_expression<operator_type, Description, detail::is_complete_description>;
    template<class Description>
    inline constexpr bool is_complete_rand_execution_v = commondx::
        is_complete_dx_execution_expression<operator_type, Description, detail::is_complete_description>::value;

} // namespace curanddx

#endif // CURANDDX_RAND_TRAITS_HPP
