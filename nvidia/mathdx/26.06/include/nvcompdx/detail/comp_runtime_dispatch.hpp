// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <cassert>
#include <utility>

namespace nvcompdx::detail::runtime {

template <template <algorithm> class Trait, typename... Args>
constexpr __host__ unsigned long long dispatch_config_trait(const algorithm A, Args&&... args) {
  switch (A) {
    case algorithm::ans:
      return Trait<algorithm::ans>::execute(std::forward<Args>(args)...);
    case algorithm::lz4:
      return Trait<algorithm::lz4>::execute(std::forward<Args>(args)...);
  }
  assert(false);
  return 0;
}

template <template <grouptype, algorithm, direction> class Trait, algorithm A, typename... Args>
constexpr __host__ unsigned long long dispatch_config_trait(const grouptype G, const direction D, Args&&... args) {
  switch (G) {
    case grouptype::warp:
      return D == direction::compress
               ? Trait<grouptype::warp, A, direction::compress>::execute(std::forward<Args>(args)...)
               : Trait<grouptype::warp, A, direction::decompress>::execute(std::forward<Args>(args)...);
    case grouptype::block:
      return D == direction::compress
               ? Trait<grouptype::block, A, direction::compress>::execute(std::forward<Args>(args)...)
               : Trait<grouptype::block, A, direction::decompress>::execute(std::forward<Args>(args)...);
  }
  assert(false);
  return 0;
}

template <template <grouptype, algorithm, direction> class Trait, typename... Args>
constexpr __host__ unsigned long long
dispatch_config_trait(const grouptype G, const algorithm A, const direction D, Args&&... args) {
  switch (A) {
    case algorithm::ans:
      return dispatch_config_trait<Trait, algorithm::ans>(G, D, std::forward<Args>(args)...);
    case algorithm::lz4:
      return dispatch_config_trait<Trait, algorithm::lz4>(G, D, std::forward<Args>(args)...);
  }
  assert(false);
  return 0;
}

template <template <grouptype, datatype, algorithm, direction> class Trait, algorithm A, datatype DT, typename... Args>
constexpr __host__ unsigned long long dispatch_config_trait(const grouptype G, const direction D, Args&&... args) {
  switch (G) {
    case grouptype::warp:
      return D == direction::compress
               ? Trait<grouptype::warp, DT, A, direction::compress>::execute(std::forward<Args>(args)...)
               : Trait<grouptype::warp, DT, A, direction::decompress>::execute(std::forward<Args>(args)...);
    case grouptype::block:
      return D == direction::compress
               ? Trait<grouptype::block, DT, A, direction::compress>::execute(std::forward<Args>(args)...)
               : Trait<grouptype::block, DT, A, direction::decompress>::execute(std::forward<Args>(args)...);
  }
  assert(false);
  return 0;
}

template <template <grouptype, datatype, algorithm, direction> class Trait, typename... Args>
constexpr __host__ unsigned long long
dispatch_config_trait(const grouptype G, const datatype DT, const algorithm A, const direction D, Args&&... args) {
  switch (A) {
    case algorithm::ans:
      switch (DT) {
        case datatype::uint8:
          return dispatch_config_trait<Trait, algorithm::ans, datatype::uint8>(G, D, std::forward<Args>(args)...);
        case datatype::float16:
          return dispatch_config_trait<Trait, algorithm::ans, datatype::float16>(G, D, std::forward<Args>(args)...);
        default:
          assert(false);
          return 0;
      }
      break;
    case algorithm::lz4:
      switch (DT) {
        case datatype::uint8:
          return dispatch_config_trait<Trait, algorithm::lz4, datatype::uint8>(G, D, std::forward<Args>(args)...);
        case datatype::uint16:
          return dispatch_config_trait<Trait, algorithm::lz4, datatype::uint16>(G, D, std::forward<Args>(args)...);
        case datatype::uint32:
          return dispatch_config_trait<Trait, algorithm::lz4, datatype::uint32>(G, D, std::forward<Args>(args)...);
        default:
          assert(false);
          return 0;
      }
      break;
  }
  assert(false);
  return 0;
}

template <template <datatype, algorithm> class Trait>
constexpr __host__ auto dispatch_config_trait(const datatype DT, const algorithm A)
  -> decltype(Trait<datatype::uint8, algorithm::ans>::execute()) {
  switch (A) {
    case algorithm::ans:
      switch (DT) {
        case datatype::uint8:
          return Trait<datatype::uint8, algorithm::ans>::execute();
        case datatype::uint16:
          return Trait<datatype::uint16, algorithm::ans>::execute();
        case datatype::uint32:
          return Trait<datatype::uint32, algorithm::ans>::execute();
        case datatype::float16:
          return Trait<datatype::float16, algorithm::ans>::execute();
      }
      break;
    case algorithm::lz4:
      switch (DT) {
        case datatype::uint8:
          return Trait<datatype::uint8, algorithm::lz4>::execute();
        case datatype::uint16:
          return Trait<datatype::uint16, algorithm::lz4>::execute();
        case datatype::uint32:
          return Trait<datatype::uint32, algorithm::lz4>::execute();
        case datatype::float16:
          return Trait<datatype::float16, algorithm::lz4>::execute();
      }
      break;
  }
  assert(false);
  return {};
}

} // namespace nvcompdx::detail::runtime
