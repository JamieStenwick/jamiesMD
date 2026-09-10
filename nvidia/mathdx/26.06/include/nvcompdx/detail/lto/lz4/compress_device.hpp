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
class InputAlignment<G, DT, algorithm::lz4, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() {
    switch (DT) {
      case datatype::uint8:
        return 1;
      case datatype::uint16:
        return 2;
      case datatype::uint32:
        return 4;
      default:
        return 1;
    }
  }
};

template <grouptype G, datatype DT>
class OutputAlignment<G, DT, algorithm::lz4, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 1; }
};

template <grouptype G, datatype DT>
class ShmemAlignment<G, DT, algorithm::lz4, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 1; }
};

template <grouptype G, datatype DT>
class TmpAlignment<G, DT, algorithm::lz4, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 2; }
};

template <datatype DT>
class ShmemSizeGroup<grouptype::block, DT, algorithm::lz4, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute([[maybe_unused]] const int warps_per_group) {
    return 0;
  }
};

template <datatype DT>
class ShmemSizeGroup<grouptype::warp, DT, algorithm::lz4, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long execute([[maybe_unused]] const int warps_per_group) {
    return 0;
  }
};

template <grouptype G>
class TmpSizeGroup<G, algorithm::lz4, direction::compress>
{
public:
  static constexpr __host__ __device__ unsigned long long
  execute(const unsigned long long max_uncomp_chunk_size, [[maybe_unused]] const datatype dt) {
    return 2 * getHashTableSize(max_uncomp_chunk_size);
  }

private:
  static constexpr __host__ __device__ unsigned long long
  getHashTableSize(const unsigned long long max_uncomp_chunk_size) {
    return detail::min(roundUpPow2(max_uncomp_chunk_size), static_cast<unsigned long long>(1U << 14));
  }
};

template <grouptype G>
class TmpSizeTotal<G, algorithm::lz4, direction::compress>
{
public:
  static constexpr unsigned long long
  execute(const unsigned long long max_uncomp_chunk_size, const datatype dt, const unsigned long long num_chunks) {
    const unsigned long long tmp_size_per_group =
      TmpSizeGroup<G, algorithm::lz4, direction::compress>().execute(max_uncomp_chunk_size, dt);
    return num_chunks * tmp_size_per_group;
  }
};

template <datatype DT>
class MinSupportedUncompChunkSize<DT, algorithm::lz4>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() {
    switch (DT) {
      case datatype::uint8:
        return 1;
      case datatype::uint16:
        return 2;
      case datatype::uint32:
        return 4;
      default:
        return 1;
    }
  }
};

template <datatype DT>
class MaxSupportedUncompChunkSize<DT, algorithm::lz4>
{
public:
  static constexpr __host__ __device__ unsigned long long execute() { return 1 << 24; }
};

template <datatype DT>
class DataTypeSupported<DT, algorithm::lz4>
{
public:
  static constexpr __host__ __device__ bool execute() {
    switch (DT) {
      case datatype::uint8:
        [[fallthrough]];
      case datatype::uint16:
        [[fallthrough]];
      case datatype::uint32:
        return true;
      default:
        return false;
    }
  }
};

template <>
class MaxCompChunkSize<algorithm::lz4>
{
public:
  static constexpr __host__ __device__ unsigned long long execute(const unsigned long long max_uncomp_chunk_size) {
    const unsigned long long expansion = max_uncomp_chunk_size + 1 + roundUpDiv(max_uncomp_chunk_size, 255);
    return roundUpTo(expansion, sizeof(unsigned long long));
  }
};

#define gen_compress(DT)                                                                                               \
  template <grouptype G>                                                                                               \
  class Compress<G, DT, algorithm::lz4, 1, false>                                                                      \
  {                                                                                                                    \
  public:                                                                                                              \
    __device__ void execute(                                                                                           \
      const void* const uncomp_chunk,                                                                                  \
      void* const comp_chunk,                                                                                          \
      const unsigned long long uncomp_chunk_size,                                                                      \
      unsigned long long* const comp_chunk_size,                                                                       \
      unsigned char* const /* shared_buffer */,                                                                        \
      unsigned char* const tmp_buffer,                                                                                 \
      const unsigned long long /* max_uncomp_chunk_size */);                                                           \
  }

gen_compress(datatype::uint8);
gen_compress(datatype::uint16);
gen_compress(datatype::uint32);

#undef gen_compress

} // namespace nvcompdx::detail
