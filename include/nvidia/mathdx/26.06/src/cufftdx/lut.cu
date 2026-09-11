// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <cufftdx.hpp>

namespace cufftdx {
    namespace database {
        namespace detail {

            using lut_fp32_type = commondx::complex<float>;
            using lut_fp64_type = commondx::complex<double>;

#ifndef CUFFTDX_DETAIL_LUT_LINKAGE
#define CUFFTDX_DETAIL_LUT_LINKAGE extern
#endif // CUFFTDX_DETAIL_LUT_LINKAGE
            #include "cufftdx/database/lut_fp32.hpp.inc"
            #include "cufftdx/database/lut_fp64.hpp.inc"
#ifdef CUFFTDX_DETAIL_LUT_LINKAGE
#undef CUFFTDX_DETAIL_LUT_LINKAGE
#endif // CUFFTDX_DETAIL_LUT_LINKAGE

        }
    }
}
