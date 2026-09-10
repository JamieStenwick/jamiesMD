// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <type_traits>
#include <tuple>

#include "nvcompdx/detail/comp_description.hpp"
#include "nvcompdx/detail/lto/execute_impl.hpp"

namespace nvcompdx::detail {

template <class... Operators>
class comp_execution
    : public comp_description<Operators...>
    , public commondx::detail::execution_description_expression
{
  using base_type = comp_description<Operators...>;
  using this_type = comp_execution<Operators...>;

public:
  /** @brief Compresses a contiguous buffer of data (one chunk)
   *
   * @param[in] input_chunk The to-be-compressed chunk
   * @param[out] output_chunk The resulting compressed chunk
   * @param[in] input_chunk_size The size of the to-be-comrpessed chunk in bytes
   * @param[out] output_chunk_size The size of the resulting compressed chunk in bytes
   * @param[in] shared_mem_buffer The shared memory scratch buffer to be used internally by the API
   * @param[in] global_mem_buffer The global memory scratch buffer to be used internally by the API
   */
  template <typename T = void>
  std::enable_if_t<base_type::is_compress_direction, T> __device__ execute(
    const void* input_chunk,
    void* output_chunk,
    const unsigned long long input_chunk_size,
    unsigned long long* output_chunk_size,
    unsigned char* shared_mem_buffer,
    unsigned char* global_mem_buffer) {
    Compress<
      base_type::this_grouptype_v,
      base_type::this_datatype_v,
      base_type::this_algorithm_v,
      (base_type::this_algorithm_v == algorithm::ans ? base_type::this_aggregate_num_warps_v : 1),
      (base_type::this_algorithm_v == algorithm::ans ? base_type::this_aggregate_complete_v : false)>()
      .execute(
        input_chunk,
        output_chunk,
        input_chunk_size,
        output_chunk_size,
        shared_mem_buffer,
        global_mem_buffer,
        base_type::this_max_uncomp_chunk_size_v);
  }

  /** @brief Decompresses a contiguous buffer of data (one chunk)
   *
   * @param[in] input_chunk The to-be-decompressed chunk
   * @param[out] output_chunk The resulting decompressed chunk
   * @param[in] input_chunk_size The size of the compressed chunk in bytes
   * @param[out] output_chunk_size The size of the resulting decompressed chunk in bytes
   * @param[in] shared_mem_buffer The shared memory scratch buffer to be used internally by the API
   * @param[in] global_mem_buffer The global memory scratch buffer to be used internally by the API
   */
  template <typename T = void>
  std::enable_if_t<!base_type::is_compress_direction, T> __device__ execute(
    const void* input_chunk,
    void* output_chunk,
    const unsigned long long input_chunk_size,
    unsigned long long* const output_chunk_size,
    unsigned char* shared_mem_buffer,
    unsigned char* global_mem_buffer) {
    Decompress<
      base_type::this_grouptype_v,
      base_type::this_datatype_v,
      base_type::this_algorithm_v,
      (base_type::this_algorithm_v == algorithm::ans ? base_type::this_aggregate_num_warps_v : 1),
      (base_type::this_algorithm_v == algorithm::ans ? base_type::this_aggregate_complete_v : false)>()
      .execute(input_chunk, output_chunk, input_chunk_size, output_chunk_size, shared_mem_buffer, global_mem_buffer);
  }

  /** @brief Returns the maximum compressed chunk size
   *
   * @return The maximum compressed chunk size in bytes
   */
  template <typename T = unsigned long long>
  static constexpr __device__ __host__ std::enable_if_t<base_type::is_compress_direction, T> max_comp_chunk_size() {
    return MaxCompChunkSize<base_type::this_algorithm_v>::execute(base_type::this_max_uncomp_chunk_size_v);
  }

  /** @brief Returns the alignment necessary for the shared memory scratch allocation.
   *
   * @return The shared memory alignment requirement in bytes
   */
  static constexpr __device__ __host__ unsigned long long shmem_alignment() {
    return ShmemAlignment<
      base_type::this_grouptype_v,
      base_type::this_datatype_v,
      base_type::this_algorithm_v,
      base_type::this_direction_v>::execute();
  }

  /** @brief Returns the alignment necessary for the input data. Depending on the
   *         direction, this can mean either the uncompressed/raw buffer (compressor) or
   *         the compressed buffer (decompressor).
   *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
   *         satisfy this requirement, and no manual alignment is necessary.
   *
   * @return The input alignment requirement in bytes
   */
  static constexpr __device__ __host__ unsigned long long input_alignment() {
    return InputAlignment<
      base_type::this_grouptype_v,
      base_type::this_datatype_v,
      base_type::this_algorithm_v,
      base_type::this_direction_v>::execute();
  }

  /** @brief Returns the alignment necessary for the output data. Depending on the
   *         direction, this can mean either the compressed buffer (compressor) or
   *         the decompressed buffer (decompressor).
   *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
   *         satisfy this requirement, and no manual alignment is necessary.
   *
   * @return The output alignment requirement in bytes
   */
  static constexpr __device__ __host__ unsigned long long output_alignment() {
    return OutputAlignment<
      base_type::this_grouptype_v,
      base_type::this_datatype_v,
      base_type::this_algorithm_v,
      base_type::this_direction_v>::execute();
  }

  /** @brief Returns the alignment necessary for the global memory scratch allocation.
   *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
   *         satisfy this requirement, and no manual alignment is necessary.
   *
   * @return The global memory scratch alignment requirement in bytes
   */
  static constexpr __device__ __host__ unsigned long long tmp_alignment() {
    return TmpAlignment<
      base_type::this_grouptype_v,
      base_type::this_datatype_v,
      base_type::this_algorithm_v,
      base_type::this_direction_v>::execute();
  }

  /** @brief Returns the global memory scratch space allocation needed for the whole kernel.
   *
   * @param[in] num_chunks The total number of chunks processed by all API invocations
   *
   * @return The required total global memory scratch space size in bytes
   */
  static constexpr unsigned long long tmp_size_total(unsigned long long num_chunks) {
    return TmpSizeTotal<base_type::this_grouptype_v, base_type::this_algorithm_v, base_type::this_direction_v>::execute(
      base_type::this_max_uncomp_chunk_size_v,
      base_type::this_datatype_v,
      num_chunks);
  }

  /** @brief Returns the global memory scratch space needed for a single instance of execution level (one warp or one block).
   *         It is not the same as `tmp_size_total`, because there could be multiple API invocations per kernel,
   *         each requiring part of the total global memory scratch space.
   *         This API call may be useful within kernels, where multiple chunks are processed by the
   *         same thread block.
   *
   * @return The required global memory scratch space for one warp or one block in bytes
   */
  static constexpr __device__ __host__ unsigned long long tmp_size_group() {
    return TmpSizeGroup<base_type::this_grouptype_v, base_type::this_algorithm_v, base_type::this_direction_v>::execute(
      base_type::this_max_uncomp_chunk_size_v,
      base_type::this_datatype_v);
  }

  /** @brief Returns the shared memory scratch space needed for a single instance of execution level (one warp or one block).
   *
   * @return The required shared memory scratch space for one warp or one block in bytes
   */
  static constexpr __device__ __host__ unsigned long long shmem_size_group() {
    return ShmemSizeGroup<
      base_type::this_grouptype_v,
      base_type::this_datatype_v,
      base_type::this_algorithm_v,
      base_type::this_direction_v>::execute(base_type::this_aggregate_num_warps_v);
  }

protected:
  /// ---- Constraints
  // Either warp or block execution must be chosen
  static_assert(
    !(base_type::has_warp && base_type::has_block),
    "Can't create nvcompdx function with two execution operators.");
};

template <class... Operators>
class warp_execution : public comp_execution<Operators...>
{
  using this_type = warp_execution<Operators...>;
  using base_type = comp_execution<Operators...>;

  /// ---- Constraints
  static_assert(!(base_type::has_block), "Can't create nvcompdx warp execution with the Block<> operator.");
  static_assert(base_type::has_warp, "Can't create nvcompdx warp execution without the Warp<> operator.");
};

template <class... Operators>
class block_execution : public comp_execution<Operators...>
{
  using this_type = block_execution<Operators...>;
  using base_type = comp_execution<Operators...>;

  /// ---- Constraints
  static_assert(!(base_type::has_warp), "Can't create nvcompdx block execution with the Warp<> operator.");
  static_assert(base_type::has_block, "Can't create nvcompdx block execution without the Block<> operator.");
  static_assert(
    base_type::has_block_warp || base_type::has_block_dim,
    "Can't create nvcompdx block execution without the BlockDim<> or BlockWarp<> operator.");
};

} // namespace nvcompdx::detail
