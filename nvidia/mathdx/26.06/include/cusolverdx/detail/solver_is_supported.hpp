// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_IS_SUPPORTED_HPP
#define CUSOLVERDX_DETAIL_IS_SUPPORTED_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/device_info.hpp"

#include "cusolverdx/operators.hpp"
#include "cusolverdx/traits.hpp"

namespace cusolverdx {
    namespace detail {

        template<class Precision,
                 type         Type,
                 unsigned int ASize,
                 unsigned int BSize      = 0,
                 unsigned int CSize      = 0,
                 unsigned int ExtraSize  = 0,
                 unsigned int ExtraBytes = 0,
                 unsigned int Arch       = 800>
        struct is_supported_shared_size_block_execution {
            static constexpr unsigned int type_multiplier  = (Type == cusolverdx::type::real) ? 1 : 2;
            static constexpr size_t required_shared_memory = sizeof(typename Precision::a_type) * type_multiplier * (ASize) +
                                                             sizeof(typename Precision::a_type) * type_multiplier * (BSize) +
                                                             sizeof(typename Precision::a_type) * type_multiplier * (CSize) +
                                                             sizeof(typename Precision::a_type) * type_multiplier * (ExtraSize) +
                                                             ExtraBytes;
            static constexpr bool value = (required_shared_memory <= commondx::device_info<Arch>::shared_memory());
        };

    } // namespace detail

    // Check if a description is supported on a given CUDA architecture
    template<class Description, unsigned int Arch>
    struct is_supported:
        public COMMONDX_STL_NAMESPACE::bool_constant<
            is_thread_v<Description> || is_cluster_v<Description> ||
            detail::is_supported_shared_size_block_execution<precision_of<Description>,
                                                             type_of_v<Description>,
                                                             matrix_size_of_v_a<Description>,
                                                             matrix_size_of_v_b<Description>,
                                                             matrix_size_of_v_c<Description>,
                                                             matrix_size_of_v_extra_size<Description>,
                                                             matrix_size_of_v_extra_bytes<Description>,
                                                             Arch>::value> {};

    template<class Description, unsigned int Arch>
    inline constexpr bool is_supported_v = is_supported<Description, Arch>::value;
} // namespace cusolverdx

#endif // CUDescriptionDX_DETAIL_IS_SUPPORTED_HPP
