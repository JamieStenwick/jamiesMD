// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "nvcompdx/types.hpp"

namespace nvcompdx::detail {

template <grouptype G, datatype DT, algorithm A, direction D>
class ShmemSizeGroup;

template <grouptype G, algorithm A, direction D>
class TmpSizeTotal;

template <grouptype G, algorithm A, direction D>
class TmpSizeGroup;

template <grouptype G, datatype DT, algorithm A, direction D>
class ShmemAlignment;

template <grouptype G, datatype DT, algorithm A, direction D>
class InputAlignment;

template <grouptype G, datatype DT, algorithm A, direction D>
class OutputAlignment;

template <grouptype G, datatype DT, algorithm A, direction D>
class TmpAlignment;

template <datatype DT, algorithm A>
class MinSupportedUncompChunkSize;

template <datatype DT, algorithm A>
class MaxSupportedUncompChunkSize;

template <datatype DT, algorithm A>
class DataTypeSupported;

template <algorithm A>
class MaxCompChunkSize;

template <grouptype G, datatype DT, algorithm A, unsigned int NumWarps, bool Complete>
class Compress;

template <grouptype G, datatype DT, algorithm A, unsigned int NumWarps, bool Complete>
class Decompress;

} // namespace nvcompdx::detail
