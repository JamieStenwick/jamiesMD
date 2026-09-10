// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "nvcompdx/detail/lto/common_fd.hpp"

namespace nvcompdx::detail {

template <algorithm A>
struct dependent_false_algorithm : detail::std::false_type {
};

template <grouptype G, datatype DT, algorithm A, unsigned int NumWarps, bool Complete>
class Compress
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in compress(). Please consult "
    "the selected compression algorithm's documentation for a description of "
    "the accepted configurations.");

  __device__ void execute(
    const void* const uncomp_chunk,
    void* const comp_chunk,
    const unsigned long long uncomp_chunk_size,
    unsigned long long* const comp_chunk_size,
    unsigned char* const shared_buffer,
    unsigned char* const tmp_buffer,
    const unsigned long long max_uncomp_chunk_size);
};

template <grouptype G, datatype DT, algorithm A, unsigned int NumWarps, bool Complete>
class Decompress
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in decompress(). Please consult "
    "the selected compression algorithm's documentation for a description of "
    "the accepted configurations.");

  __device__ void execute(
    const void* const comp_chunk,
    void* const uncomp_chunk,
    const unsigned long long comp_chunk_size,
    unsigned long long* const uncomp_chunk_size,
    unsigned char* const shared_buffer,
    unsigned char* const tmp_buffer);
};

template <grouptype G, datatype DT, algorithm A, direction D>
class ShmemSizeGroup
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in shmem_size_group(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr __host__ __device__ unsigned long long execute(const int warps_per_group) {
    // This class will be overridden, return value doesn't matter
    return 0;
  }
};

template <grouptype G, algorithm A, direction D>
class TmpSizeTotal
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in tmp_size_total(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr unsigned long long
  execute(const unsigned long long max_uncomp_chunk_size, const datatype dt, const unsigned long long num_chunks) {
    // This class will be overridden, return value doesn't matter
    return 0;
  }
};

template <grouptype G, algorithm A, direction D>
class TmpSizeGroup
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in tmp_size_group(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr unsigned long long __host__ __device__
  execute(const unsigned long long max_uncomp_chunk_size, const datatype dt) {
    // This class will be overridden, return value doesn't matter
    return 0;
  }
};

template <grouptype G, datatype DT, algorithm A, direction D>
class InputAlignment
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in input_alignment(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr unsigned long long __host__ __device__ execute() {
    // This class will be overridden, return value doesn't matter
    return 1;
  }
};

template <grouptype G, datatype DT, algorithm A, direction D>
class OutputAlignment
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in output_alignment(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr unsigned long long __host__ __device__ execute() {
    // This class will be overridden, return value doesn't matter
    return 1;
  }
};

template <grouptype G, datatype DT, algorithm A, direction D>
class ShmemAlignment
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in shmem_alignment(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr unsigned long long __host__ __device__ execute() {
    // This class will be overridden, return value doesn't matter
    return 1;
  }
};

template <grouptype G, datatype DT, algorithm A, direction D>
class TmpAlignment
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in tmp_alignment(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr unsigned long long __host__ __device__ execute() {
    // This class will be overridden, return value doesn't matter
    return 1;
  }
};

template <datatype DT, algorithm A>
class MinSupportedUncompChunkSize
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in min_uncomp_chunk_size(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr __host__ __device__ unsigned long long execute() {
    // This class will be overridden, return value doesn't matter
    return 0;
  }
};

template <datatype DT, algorithm A>
class MaxSupportedUncompChunkSize
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in max_uncomp_chunk_size(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr __host__ __device__ unsigned long long execute() {
    // This class will be overridden, return value doesn't matter
    return 0;
  }
};

template <datatype DT, algorithm A>
class DataTypeSupported
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected. Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr __host__ __device__ bool execute() {
    // This class will be overridden, return value doesn't matter
    return false;
  }
};

template <algorithm A>
class MaxCompChunkSize
{
public:
  static_assert(
    dependent_false_algorithm<A>::value,
    "Unsupported compressor config detected in max_comp_chunk_size(). Please "
    "consult the selected compression algorithm's documentation for a "
    "description of the accepted configurations.");

  static constexpr __host__ __device__ unsigned long long execute(const unsigned long long max_uncomp_chunk_size) {
    // This class will be overridden, return value doesn't matter
    return 0;
  }
};

} // namespace nvcompdx::detail
