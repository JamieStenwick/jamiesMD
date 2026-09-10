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
#include "nvcompdx/detail/comp_runtime_dispatch.hpp"

namespace nvcompdx::detail::runtime {

/** @brief Returns the maximum compressed chunk size
 *
 * @param[in] A Algorithm (ans, lz4)
 * @param[in] max_uncomp_chunk_size The maximum size of the uncompressed chunk in bytes
 *
 * @return The maximum compressed chunk size in bytes
 */
constexpr __host__ unsigned long long
max_comp_chunk_size(const algorithm A, const unsigned long long max_uncomp_chunk_size) {
  return dispatch_config_trait<MaxCompChunkSize>(A, max_uncomp_chunk_size);
}

/** @brief Returns the minimum supported uncompressed chunk size in bytes.
 *
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 *
 * @return The minimum supported uncompressed chunk size in bytes
 */
constexpr __host__ unsigned long long min_supported_uncomp_chunk_size(const datatype DT, const algorithm A) {
  return dispatch_config_trait<MinSupportedUncompChunkSize>(DT, A);
}

/** @brief Returns the maximum supported uncompressed chunk size in bytes.
 *
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 *
 * @return The maximum supported uncompressed chunk size in bytes
 */
constexpr __host__ unsigned long long max_supported_uncomp_chunk_size(const datatype DT, const algorithm A) {
  return dispatch_config_trait<MaxSupportedUncompChunkSize>(DT, A);
}

/** @brief Returns whether the given data type is supported by the algorithm.
 *
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 *
 * @return true if the data type is supported, false otherwise
 */
constexpr __host__ bool datatype_supported(const datatype DT, const algorithm A) {
  return dispatch_config_trait<DataTypeSupported>(DT, A);
}

/** @brief Returns the shared memory scratch space needed for a single instance of execution level (one warp or one block).
 *
 * @param[in] G Group type (warp or block)
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 * @param[in] D Direction (compress or decompress)
 * @param[in] warps_per_group The number of warps per group
 *
 * @return The required shared memory scratch space for one warp or one block in bytes
 */
constexpr __host__ unsigned long long
shmem_size_group(const grouptype G, const datatype DT, const algorithm A, const direction D, const int warps_per_group) {
  return dispatch_config_trait<ShmemSizeGroup>(G, DT, A, D, warps_per_group);
}

/** @brief Returns the global memory scratch space allocation needed for the whole kernel.
 *
 * @param[in] G Group type (warp or block)
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 * @param[in] D Direction (compress or decompress)
 * @param[in] max_uncomp_chunk_size The maximum size of the uncompressed chunk in bytes
 * @param[in] num_chunks The total number of chunks processed by all API invocations
 *
 * @return The required total global memory scratch space size in bytes
 */
constexpr __host__ unsigned long long tmp_size_total(
  const grouptype G,
  const datatype DT,
  const algorithm A,
  const direction D,
  const unsigned long long max_uncomp_chunk_size,
  const unsigned long long num_chunks) {
  return dispatch_config_trait<TmpSizeTotal>(G, A, D, max_uncomp_chunk_size, DT, num_chunks);
}

/** @brief Returns the global memory scratch space needed for a single instance of execution level (one warp or one block).
 *         It is not the same as `tmp_size_total`, because there could be multiple API invocations per kernel,
 *         each requiring part of the total global memory scratch space.
 *         This API call may be useful within kernels, where multiple chunks are processed by the
 *         same thread block.
 *
 * @param[in] G Group type (warp or block)
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 * @param[in] D Direction (compress or decompress)
 * @param[in] max_uncomp_chunk_size The maximum size of the uncompressed chunk in bytes
 *
 * @return The required global memory scratch space for one warp or one block in bytes
 */
constexpr __host__ unsigned long long tmp_size_group(
  const grouptype G,
  const datatype DT,
  const algorithm A,
  const direction D,
  const unsigned long long max_uncomp_chunk_size) {
  return dispatch_config_trait<TmpSizeGroup>(G, A, D, max_uncomp_chunk_size, DT);
}

/** @brief Returns the alignment necessary for the input data. Depending on the
 *         direction, this can mean either the uncompressed/raw buffer (compressor) or
 *         the compressed buffer (decompressor).
 *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
 *         satisfy this requirement, and no manual alignment is necessary.
 *
 * @param[in] G Group type (warp or block)
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 * @param[in] D Direction (compress or decompress)
 *
 * @return The input alignment requirement in bytes
 */
constexpr __host__ unsigned long long
input_alignment(const grouptype G, const datatype DT, const algorithm A, const direction D) {
  return dispatch_config_trait<InputAlignment>(G, DT, A, D);
}

/** @brief Returns the alignment necessary for the output data. Depending on the
 *         direction, this can mean either the compressed buffer (compressor) or
 *         the decompressed buffer (decompressor).
 *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
 *         satisfy this requirement, and no manual alignment is necessary.
 *
 * @param[in] G Group type (warp or block)
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 * @param[in] D Direction (compress or decompress)
 *
 * @return The output alignment requirement in bytes
 */
constexpr __host__ unsigned long long
output_alignment(const grouptype G, const datatype DT, const algorithm A, const direction D) {
  return dispatch_config_trait<OutputAlignment>(G, DT, A, D);
}

/** @brief Returns the alignment necessary for the shared memory scratch allocation.
 *
 * @param[in] G Group type (warp or block)
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 * @param[in] D Direction (compress or decompress)
 *
 * @return The shared memory alignment requirement in bytes
 */
constexpr __host__ unsigned long long
shmem_alignment(const grouptype G, const datatype DT, const algorithm A, const direction D) {
  return dispatch_config_trait<ShmemAlignment>(G, DT, A, D);
}

/** @brief Returns the alignment necessary for the global memory scratch allocation.
 *         Allocations provided by cudaMalloc() / cudaMallocPitch() / etc. automatically
 *         satisfy this requirement, and no manual alignment is necessary.
 *
 * @param[in] G Group type (warp or block)
 * @param[in] DT Data type (uint8, uint16, uint32, float16)
 * @param[in] A Algorithm (ans, lz4)
 * @param[in] D Direction (compress or decompress)
 *
 * @return The global memory scratch alignment requirement in bytes
 */
constexpr __host__ unsigned long long
tmp_alignment(const grouptype G, const datatype DT, const algorithm A, const direction D) {
  return dispatch_config_trait<TmpAlignment>(G, DT, A, D);
}

} // namespace nvcompdx::detail::runtime
