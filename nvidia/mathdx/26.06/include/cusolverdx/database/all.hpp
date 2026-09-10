// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_ALL_HPP
#define CUSOLVERDX_DATABASE_ALL_HPP

// Gaussian Elimination
#include "cusolverdx/database/potrf.cuh"
#include "cusolverdx/database/potrs.cuh"
#include "cusolverdx/database/posv.cuh"
#include "cusolverdx/database/getrf_no_pivot.cuh"
#include "cusolverdx/database/getrs_no_pivot.cuh"
#include "cusolverdx/database/gesv_no_pivot.cuh"
#include "cusolverdx/database/getrf_partial_pivot.cuh"
#include "cusolverdx/database/getrs_partial_pivot.cuh"
#include "cusolverdx/database/gesv_partial_pivot.cuh"
#include "cusolverdx/database/gtsv_no_pivot.cuh"

// QR
#include "cusolverdx/database/geqrf.cuh"
#include "cusolverdx/database/gelqf.cuh"
#include "cusolverdx/database/unmqr.cuh"
#include "cusolverdx/database/unmlq.cuh"
#include "cusolverdx/database/ungqr.cuh"
#include "cusolverdx/database/unglq.cuh"
#include "cusolverdx/database/gels.cuh"

// Symmetric eigenvalue
#include "cusolverdx/database/htev.cuh"
#include "cusolverdx/database/heev.cuh"
#include "cusolverdx/database/hegst.cuh"
#include "cusolverdx/database/hegv.cuh"

// SVD
#include "cusolverdx/database/bdsvd.cuh"
#include "cusolverdx/database/gesvd.cuh"

// BLAS
#include "cusolverdx/database/trsm.cuh"

#endif // CUSOLVERDX_DATABASE_ALL_HPP

