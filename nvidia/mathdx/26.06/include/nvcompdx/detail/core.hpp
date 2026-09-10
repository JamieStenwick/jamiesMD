// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#define NVCOMPDX_DETAIL_IGNORE_FALLBACK
#include "nvcompdx/detail/lto/execute_impl.hpp"
#undef NVCOMPDX_DETAIL_IGNORE_FALLBACK

namespace nvcompdx::detail::core {

/** @brief Compresses a contiguous buffer of data (one chunk)
 *
 * @tparam G Group type (warp or block)
 * @tparam DT Data type (uint8, uint16, uint32, float16)
 * @tparam A Algorithm (ans, lz4)
 * @tparam NumWarps Number of warps per group
 * @tparam Complete Boolean, indicating for block mode that the thread block contains an integer multiple of warps
 *
 * @param[in] input_chunk The to-be-compressed chunk
 * @param[out] output_chunk The resulting compressed chunk
 * @param[in] input_chunk_size The size of the to-be-compressed chunk in bytes
 * @param[out] output_chunk_size The size of the resulting compressed chunk in bytes
 * @param[in] shared_mem_buffer The shared memory scratch buffer to be used internally by the API
 * @param[in] global_mem_buffer The global memory scratch buffer to be used internally by the API
 * @param[in] max_uncomp_chunk_size The maximum size of the uncompressed chunk in bytes
 */
template <grouptype G, datatype DT, algorithm A, unsigned int NumWarps, bool Complete>
constexpr __host__ __device__ void compress(
  const void* input_chunk,
  void* output_chunk,
  const unsigned long long input_chunk_size,
  unsigned long long* output_chunk_size,
  unsigned char* shared_mem_buffer,
  unsigned char* global_mem_buffer,
  const unsigned long long max_uncomp_chunk_size) {
  static_assert(NumWarps >= 1, "Number of warps in the thread block must be greater than or equal to 1");
  return Compress<G, DT, A, (A == algorithm::ans ? NumWarps : 1), (A == algorithm::ans ? Complete : false)>().execute(
    input_chunk,
    output_chunk,
    input_chunk_size,
    output_chunk_size,
    shared_mem_buffer,
    global_mem_buffer,
    max_uncomp_chunk_size);
}

/** @brief Decompresses a contiguous buffer of data (one chunk)
 *
 * @tparam G Group type (warp or block)
 * @tparam DT Data type (uint8, uint16, uint32, float16)
 * @tparam A Algorithm (ans, lz4)
 * @tparam NumWarps Number of warps per group
 * @tparam Complete Boolean, indicating for block mode that the thread block contains an integer multiple of warps
 *
 * @param[in] input_chunk The to-be-decompressed chunk
 * @param[out] output_chunk The resulting decompressed chunk
 * @param[in] input_chunk_size The size of the compressed chunk in bytes
 * @param[out] output_chunk_size The size of the resulting decompressed chunk in bytes
 * @param[in] shared_mem_buffer The shared memory scratch buffer to be used internally by the API
 * @param[in] global_mem_buffer The global memory scratch buffer to be used internally by the API
 */
template <grouptype G, datatype DT, algorithm A, unsigned int NumWarps, bool Complete>
constexpr __host__ __device__ void decompress(
  const void* input_chunk,
  void* output_chunk,
  const unsigned long long input_chunk_size,
  unsigned long long* output_chunk_size,
  unsigned char* shared_mem_buffer,
  unsigned char* global_mem_buffer) {
  static_assert(NumWarps >= 1, "Number of warps in the thread block must be greater than or equal to 1");
  return Decompress<G, DT, A, (A == algorithm::ans ? NumWarps : 1), (A == algorithm::ans ? Complete : false)>()
    .execute(input_chunk, output_chunk, input_chunk_size, output_chunk_size, shared_mem_buffer, global_mem_buffer);
}

/** @brief Returns the maximum compressed chunk size
 *
 * @tparam A Algorithm (ans, lz4)
 *
 * @param[in] max_uncomp_chunk_size The maximum size of the uncompressed chunk in bytes
 *
 * @return The maximum compressed chunk size in bytes
 */
template <algorithm A>
constexpr __host__ __device__ unsigned long long max_comp_chunk_size(unsigned long long max_uncomp_chunk_size) {
  return nvcompdx::detail::MaxCompChunkSize<A>::execute(max_uncomp_chunk_size);
}

/** @brief Returns the shared memory scratch space needed for a single instance of execution level (one warp or one block).
 *
 * @tparam G Group type (warp or block)
 * @tparam DT Data type (uint8, uint16, uint32, float16)
 * @tparam A Algorithm (ans, lz4)
 * @tparam D Direction (compress or decompress)
 *
 * @param[in] warps_per_group The number of warps per group
 *
 * @return The required shared memory scratch space for one warp or one block in bytes
 */
template <grouptype G, datatype DT, algorithm A, direction D>
constexpr __host__ __device__ unsigned long long shmem_size_group(const int warps_per_group) {
  return ShmemSizeGroup<G, DT, A, D>::execute(warps_per_group);
}

/** @brief Returns the global memory scratch space allocation needed for the whole kernel.
 *
 * @tparam G Group type (warp or block)
 * @tparam A Algorithm (ans, lz4)
 * @tparam D Direction (compress or decompress)
 *
 * @param[in] max_uncomp_chunk_size The maximum size of the uncompressed chunk in bytes
 * @param[in] dt The datatype of the uncompressed chunk
 * @param[in] num_chunks The total number of chunks processed by all API invocations
 *
 * @return The required total global memory scratch space size in bytes
 */
template <grouptype G, algorithm A, direction D>
constexpr __host__ __device__ unsigned long long
tmp_size_total(const unsigned long long max_uncomp_chunk_size, const datatype dt, const unsigned long long num_chunks) {
  return TmpSizeTotal<G, A, D>::execute(max_uncomp_chunk_size, dt, num_chunks);
}

/** @brief Returns the global memory scratch space needed for a single instance of execution level (one warp or one block).
 *         It is not the same as `tmp_size_total`, because there could be multiple API invocations per kernel,
 *         each requiring part of the total global memory scratch space.
 *         This API call may be useful within kernels, where multiple chunks are processed by the
 *         same thread block.
 *
 * @tparam G Group type (warp or block)
 * @tparam A Algorithm (ans, lz4)
 * @tparam D Direction (compress or decompress)
 *
 * @param[in] max_uncomp_chunk_size The maximum size of the uncompressed chunk in bytes
 * @param[in] dt The datatype of the uncompressed chunk
 *
 * @return The required global memory scratch space for one warp or one block in bytes
 */
template <grouptype G, algorithm A, direction D>
constexpr __host__ __device__ unsigned long long
tmp_size_group(const unsigned long long max_uncomp_chunk_size, const datatype dt) {
  return TmpSizeGroup<G, A, D>::execute(max_uncomp_chunk_size, dt);
}

/** @brief Returns the alignment necessary for the input data. Depending on the
 *         direction, this can mean either the uncompressed/raw buffer (compressor) or
 *         the compressed buffer (decompressor).
 *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
 *         satisfy this requirement, and no manual alignment is necessary.
 *
 * @tparam G Group type (warp or block)
 * @tparam DT Data type (uint8, uint16, uint32, float16)
 * @tparam A Algorithm (ans, lz4)
 * @tparam D Direction (compress or decompress)
 *
 * @return The input alignment requirement in bytes
 */
template <grouptype G, datatype DT, algorithm A, direction D>
constexpr __host__ __device__ unsigned long long input_alignment() {
  return nvcompdx::detail::InputAlignment<G, DT, A, D>::execute();
}

/** @brief Returns the alignment necessary for the output data. Depending on the
 *         direction, this can mean either the compressed buffer (compressor) or
 *         the decompressed buffer (decompressor).
 *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
 *         satisfy this requirement, and no manual alignment is necessary.
 *
 * @tparam G Group type (warp or block)
 * @tparam DT Data type (uint8, uint16, uint32, float16)
 * @tparam A Algorithm (ans, lz4)
 * @tparam D Direction (compress or decompress)
 *
 * @return The output alignment requirement in bytes
 */
template <grouptype G, datatype DT, algorithm A, direction D>
constexpr __host__ __device__ unsigned long long output_alignment() {
  return nvcompdx::detail::OutputAlignment<G, DT, A, D>::execute();
}

/** @brief Returns the alignment necessary for the shared memory scratch allocation.
 *
 * @tparam G Group type (warp or block)
 * @tparam DT Data type (uint8, uint16, uint32, float16)
 * @tparam A Algorithm (ans, lz4)
 * @tparam D Direction (compress or decompress)
 *
 * @return The shared memory alignment requirement in bytes
 */
template <grouptype G, datatype DT, algorithm A, direction D>
constexpr __host__ __device__ unsigned long long shmem_alignment() {
  return nvcompdx::detail::ShmemAlignment<G, DT, A, D>::execute();
}

/** @brief Returns the alignment necessary for the global memory scratch allocation.
 *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
 *         satisfy this requirement, and no manual alignment is necessary.
 *
 * @tparam G Group type (warp or block)
 * @tparam DT Data type (uint8, uint16, uint32, float16)
 * @tparam A Algorithm (ans, lz4)
 * @tparam D Direction (compress or decompress)
 *
 * @return The global memory scratch alignment requirement in bytes
 */
template <grouptype G, datatype DT, algorithm A, direction D>
constexpr __host__ __device__ unsigned long long tmp_alignment() {
  return nvcompdx::detail::TmpAlignment<G, DT, A, D>::execute();
}

} // namespace nvcompdx::detail::core
