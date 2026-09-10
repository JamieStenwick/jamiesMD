// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

namespace nvcompdx::detail {

/**
 * @enum grouptype
 *
 * @brief Execution group type that the compression or decompression operates with.
 */
enum class grouptype
{
  /// The internal group of threads working on a single chunk will be a single warp
  warp,

  /// The internal group of threads working on a single chunk will be a thread block
  block
};

} // namespace nvcompdx::detail
