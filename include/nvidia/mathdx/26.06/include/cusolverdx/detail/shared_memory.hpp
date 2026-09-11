// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_SHARED_MEMORY_HPP
#define CUSOLVERDX_DETAIL_SHARED_MEMORY_HPP

#include "commondx/shared_memory.hpp"

namespace cusolverdx {
    namespace shared_memory {
        using commondx::shared_memory::slice_into_pointers;
        using commondx::shared_memory::slice_into_tensors;
        using commondx::shared_memory::slice;
    } // namespace shared_memory

    using commondx::make_shared_storage_calculator;
} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_SHARED_MEMORY_HPP
