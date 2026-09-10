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
class InputAlignment<G, DT, algorithm::ans, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 16; }
};

template <grouptype G, datatype DT>
class OutputAlignment<G, DT, algorithm::ans, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 8; }
};

template <grouptype G, datatype DT>
class ShmemAlignment<G, DT, algorithm::ans, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 16; }
};

template <grouptype G, datatype DT>
class TmpAlignment<G, DT, algorithm::ans, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 2; }
};

template <datatype DT>
class ShmemSizeGroup<grouptype::block, DT, algorithm::ans, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute(const int warps_per_group) {
    if (warps_per_group == 1) {
      return 3584;
    }

    const unsigned long long shmem_size_per_block = detail::max(5324, 3072 + warps_per_group * 64);
    switch (DT) {
      case datatype::uint8:
        return shmem_size_per_block;
      case datatype::float16:
        return detail::max(shmem_size_per_block, 1024ull + warps_per_group * 256);
      default:
        return 0;
    }
  }
};

template <datatype DT>
class ShmemSizeGroup<grouptype::warp, DT, algorithm::ans, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute([[maybe_unused]] const int warps_per_group) {
    return roundUpTo(3584ull, ShmemAlignment<grouptype::warp, DT, algorithm::ans, direction::compress>::execute());
  }
};

template <grouptype G>
class TmpSizeTotal<G, algorithm::ans, direction::compress>
{
public:
  static constexpr unsigned long long
  execute(const unsigned long long max_uncomp_chunk_size, const datatype dt, const unsigned long long num_chunks) {
    // Here, we need half of the chunk to store all the exponents
    const unsigned long long tmp_size_per_group = max_uncomp_chunk_size / 2 + 1024;

    switch (dt) {
      case datatype::uint8:
        return 0;
      case datatype::float16:
        return num_chunks * tmp_size_per_group;
      default:
        return 0;
    }
  }
};

template <grouptype G>
class TmpSizeGroup<G, algorithm::ans, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long
  execute(const unsigned long long max_uncomp_chunk_size, const datatype dt) {
    // Here, we need half of the chunk to store all the exponents
    const unsigned long long tmp_size_per_group = max_uncomp_chunk_size / 2 + 1024;

    switch (dt) {
      case datatype::uint8:
        return 0;
      case datatype::float16:
        return tmp_size_per_group;
      default:
        return 0;
    }
  }
};

template <datatype DT>
class MinSupportedUncompChunkSize<DT, algorithm::ans>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() {
    switch (DT) {
      case datatype::uint8:
        return 1;
      case datatype::float16:
        return 2;
      default:
        return 1;
    }
  }
};

template <datatype DT>
class MaxSupportedUncompChunkSize<DT, algorithm::ans>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 1 << 18; }
};

template <datatype DT>
class DataTypeSupported<DT, algorithm::ans>
{
public:
  static constexpr __host__ __device__ bool execute() {
    switch (DT) {
      case datatype::uint8:
        [[fallthrough]];
      case datatype::float16:
        return true;
      default:
        return false;
    }
  }
};

template <>
class MaxCompChunkSize<algorithm::ans>
{
public:
  static constexpr __host__ __device__ unsigned long long execute(const unsigned long long max_uncomp_chunk_size) {
    /* Assuming a tablelog of 10, a maximum average of 10 bits can be written out for each
     * 8-bit symbol. So the maximum overhead is about 10/8 = 1.2. We add a small safety overhead
     * 0.1 and a constant offset of 1576, which is the largest header we can have.
     */
    return static_cast<unsigned long long>(1.3 * max_uncomp_chunk_size) + 1576;
  }
};

#define gen_compress(DT)                                                                                               \
  template <grouptype G, unsigned int NumWarps, bool Complete>                                                         \
  class Compress<G, DT, algorithm::ans, NumWarps, Complete>                                                            \
  {                                                                                                                    \
  public:                                                                                                              \
    __device__ void execute(                                                                                           \
      const void* const uncomp_chunk,                                                                                  \
      void* const comp_chunk,                                                                                          \
      const unsigned long long uncomp_chunk_size,                                                                      \
      unsigned long long* const comp_chunk_size,                                                                       \
      unsigned char* const shared_buffer,                                                                              \
      unsigned char* const tmp_buffer,                                                                                 \
      const unsigned long long max_uncomp_chunk_size);                                                                 \
  }

gen_compress(datatype::uint8);
gen_compress(datatype::float16);

#undef gen_compress

} // namespace nvcompdx::detail
