// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "nvcompdx/detail/lto/types.hpp"
#include "nvcompdx/operators/operator_type.hpp"

namespace nvcompdx {

/**
 * @enum direction
 *
 * @brief Direction of the intended operation.
 */
enum class direction
{
  compress,
  decompress
};

/**
 * @enum algorithm
 *
 * @brief The compression algorithm to be used.
 */
enum class algorithm
{
  ans,
  lz4,
};

/**
 * @enum datatype
 *
 * @brief The way in which the compression algorithm will interpret the input data.
 */
enum class datatype
{
  /// Data to be interpreted as consecutive bytes. If the input datatype is not included
  /// in the options below, uint8 should be selected.
  uint8,

  /// Data to be interpreted as consecutive shorts (2 bytes).
  /// Requires the total number of input bytes per chunk to be divisible by two.
  uint16,

  /// Data to be interpreted as consecutive integers (4 bytes).
  /// Requires the total number of input bytes per chunk to be divisible by four.
  uint32,

  /// Data to be interpreted as consecutive half-precision floats (2 bytes).
  /// Requires the total number of input bytes per chunk to be divisible by two.
  float16
};

} // namespace nvcompdx
