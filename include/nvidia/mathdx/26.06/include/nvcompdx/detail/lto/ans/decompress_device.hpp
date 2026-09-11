// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "nvcompdx/detail/lto/common_fd.hpp"
#include "nvcompdx/detail/utils.hpp"

namespace nvcompdx::detail {

template <grouptype G, datatype DT>
class InputAlignment<G, DT, algorithm::ans, direction::decompress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 8; }
};

template <grouptype G, datatype DT>
class OutputAlignment<G, DT, algorithm::ans, direction::decompress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 1; }
};

template <grouptype G, datatype DT>
class ShmemAlignment<G, DT, algorithm::ans, direction::decompress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 2; }
};

template <grouptype G, datatype DT>
class TmpAlignment<G, DT, algorithm::ans, direction::decompress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 1; }
};

template <datatype DT>
class ShmemSizeGroup<grouptype::block, DT, algorithm::ans, direction::decompress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute(const int warps_per_group) {
    return detail::max(2050, 1538 + 256 * warps_per_group);
  }
};

template <datatype DT>
class ShmemSizeGroup<grouptype::warp, DT, algorithm::ans, direction::decompress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute([[maybe_unused]] const int warps_per_group) {
    return roundUpTo(2050ull, ShmemAlignment<grouptype::warp, DT, algorithm::ans, direction::decompress>::execute());
  }
};

template <grouptype G>
class TmpSizeTotal<G, algorithm::ans, direction::decompress>
{
public:
  static constexpr unsigned long long execute(
    [[maybe_unused]] const unsigned long long max_uncomp_chunk_size,
    [[maybe_unused]] const datatype dt,
    [[maybe_unused]] const unsigned long long num_chunks) {
    return 0;
  }
};

template <grouptype G>
class TmpSizeGroup<G, algorithm::ans, direction::decompress>
{
public:
  static constexpr __host__ __device__ unsigned long long
  execute([[maybe_unused]] const unsigned long long max_uncomp_chunk_size, [[maybe_unused]] const datatype dt) {
    return 0;
  }
};

#define gen_decompress(DT)                                                                                             \
  template <grouptype G, unsigned int NumWarps, bool Complete>                                                         \
  class Decompress<G, DT, algorithm::ans, NumWarps, Complete>                                                          \
  {                                                                                                                    \
  public:                                                                                                              \
    __device__ void execute(                                                                                           \
      const void* const comp_chunk,                                                                                    \
      void* const uncomp_chunk,                                                                                        \
      const unsigned long long /* comp_chunk_size */,                                                                  \
      unsigned long long* const uncomp_chunk_size,                                                                     \
      unsigned char* const shared_buffer,                                                                              \
      unsigned char* const /* tmp_buffer */);                                                                          \
  }

gen_decompress(datatype::uint8);
gen_decompress(datatype::float16);

#undef gen_decompress

} // namespace nvcompdx::detail
