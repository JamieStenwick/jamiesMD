// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <type_traits>

#include "commondx/device_info.hpp"

#include "nvcompdx/operators.hpp"
#include "nvcompdx/types.hpp"
#include "nvcompdx/detail/lto/common_fd.hpp"

namespace nvcompdx::detail {
// Is the specified datatype supported in the algorithm?
template <datatype DT, algorithm A>
struct is_supported_datatype {
  static constexpr bool value = DataTypeSupported<DT, A>::execute();
};

// Is the specified maximum uncompressed chunk size supported?
template <datatype DT, algorithm A, unsigned long long max_uncomp_chunk_size>
struct is_supported_max_uncomp_chunk_size {
  static constexpr unsigned long long min_value = MinSupportedUncompChunkSize<DT, A>::execute();
  static constexpr unsigned long long max_value = MaxSupportedUncompChunkSize<DT, A>::execute();
  static constexpr bool value = max_uncomp_chunk_size >= min_value && max_uncomp_chunk_size <= max_value;
};

// Is the specified problem description supported on the given SM?
// Note: we are only checking the shared scratch space eligibility, and not additional
//       buffer space required for holding compressed/decompressed data in shared memory
template <datatype DT, algorithm A, direction D, grouptype G, int num_warps, unsigned int Arch>
struct is_supported_shared_size {
  static constexpr unsigned long long required_bytes = ShmemSizeGroup<G, DT, A, D>::execute(num_warps);
  static constexpr bool value = required_bytes <= commondx::device_info<Arch>::shared_memory();
};
} // namespace nvcompdx::detail
