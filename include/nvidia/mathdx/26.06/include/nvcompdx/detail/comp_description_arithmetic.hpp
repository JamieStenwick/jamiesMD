// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "nvcompdx/detail/comp_execution.hpp"

namespace nvcompdx {
namespace detail {

template <class... Operators>
struct make_description {
private:
  using operator_wrapper_type = comp_operator_wrapper<Operators...>;
  static constexpr bool has_warp_operator = has_operator<operator_type::warp, operator_wrapper_type>::value;
  static constexpr bool has_block_operator = has_operator<operator_type::block, operator_wrapper_type>::value;
  static constexpr bool has_block_dim_operator = has_operator<operator_type::block_dim, operator_wrapper_type>::value;
  static constexpr bool has_block_warp_operator = has_operator<operator_type::block_warp, operator_wrapper_type>::value;

  static constexpr bool has_execution_operator =
    has_warp_operator || (has_block_operator && (has_block_dim_operator || has_block_warp_operator));

  // Workaround (NVRTC/MSVC)
  //
  // For NVRTC we need to utilize an in-between class called comp_execution, otherwise
  // we run into a complation error if Block() is added to description before description is
  // complete, example:
  //
  // Fails on NVRTC:
  //     Size<...>() + Function<...>() + Type<...>() + Block() + SM<750>()
  // Works on NVRTC:
  //     Size<...>() + Function<...>() + Type<...>() + SM<750>() + Block()
  //
  // This workaround disables some useful diagnostics based on static_asserts.
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
  using block_execution_type = detail::std::conditional_t<
    is_complete_comp<operator_wrapper_type>::value,
    block_execution<Operators...>,
    comp_execution<Operators...>>;
  using warp_execution_type = detail::std::conditional_t<
    is_complete_comp<operator_wrapper_type>::value,
    warp_execution<Operators...>,
    comp_execution<Operators...>>;
#else
  using block_execution_type = block_execution<Operators...>;
  using warp_execution_type = warp_execution<Operators...>;
#endif
  using description_type = comp_description<Operators...>;
  using execution_type = detail::std::conditional_t<has_block_operator, block_execution_type, warp_execution_type>;

public:
  using type = typename detail::std::conditional<has_execution_operator, execution_type, description_type>::type;
};

template <class... Operators>
using make_description_t = typename make_description<Operators...>::type;
} // namespace detail

template <class Operator1, class Operator2>
__host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) -> detail::std::enable_if_t<
  commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
  detail::make_description_t<Operator1, Operator2>> {
  return detail::make_description_t<Operator1, Operator2>();
}

template <class... Operators1, class Operator2>
__host__ __device__ __forceinline__ auto operator+(const detail::comp_description<Operators1...>&, const Operator2&)
  -> detail::std::enable_if_t<
    commondx::detail::is_operator_expression<Operator2>::value,
    detail::make_description_t<Operators1..., Operator2>> {
  return detail::make_description_t<Operators1..., Operator2>();
}

template <class Operator1, class... Operators2>
__host__ __device__ __forceinline__ auto operator+(const Operator1&, const detail::comp_description<Operators2...>&)
  -> detail::std::enable_if_t<
    commondx::detail::is_operator_expression<Operator1>::value,
    detail::make_description_t<Operator1, Operators2...>> {
  return detail::make_description_t<Operator1, Operators2...>();
}

template <class... Operators1, class... Operators2>
__host__ __device__ __forceinline__ auto
operator+(const detail::comp_description<Operators1...>&, const detail::comp_description<Operators2...>&)
  -> detail::make_description_t<Operators1..., Operators2...> {
  return detail::make_description_t<Operators1..., Operators2...>();
}

} // namespace nvcompdx
