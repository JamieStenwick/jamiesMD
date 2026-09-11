// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_OPERATORS_ENUMS_HPP
#define CUSOLVERDX_OPERATORS_ENUMS_HPP

// The enums used by the Operator types
namespace cusolverdx {

    enum class arrangement
    {
        col_major,
        row_major,
    };
    inline constexpr auto col_major    = arrangement::col_major;
    inline constexpr auto left_layout  = arrangement::col_major;
    inline constexpr auto row_major    = arrangement::row_major;
    inline constexpr auto right_layout = arrangement::row_major;

    enum class diag
    {
        non_unit,
        unit,
    };

    enum class fill_mode
    {
        upper,
        lower,
    };
    inline constexpr auto upper = fill_mode::upper;
    inline constexpr auto lower = fill_mode::lower;

    enum class function
    {
        // Gaussian Elimination
        potrf,               // Cholesky Factorization
        potrs,               // Cholesky Solve
        posv,                // Cholesky Factor and Solve
        getrf_no_pivot,      // LU Factorization
        getrs_no_pivot,      // LU Solve
        gesv_no_pivot,       // LU Factor and Solve
        getrf_partial_pivot, // LU Factorization
        getrs_partial_pivot, // LU Solve
        gesv_partial_pivot,  // LU Factor and Solve
        gtsv_no_pivot,       // Tridiagonal Solve
        modified_lu,         // Modified LU Factorization for unitary matrix

        // QR
        geqrf,               // QR factorization
        unmqr,               // Apply QR householder factors
        ungqr,               // Generate Q from geqrf
        gelqf,               // LQ factorization
        unmlq,               // Apply LQ householder factors
        unglq,               // Generate Q from gelqf
        gels,                // Solve least squares problem with qr or lq

        // Symmetric eigenvalue
        htev,                // Eigensolver for a symmetric tridiagonal matrix
        heev,                // Eigensolver for a symmetric matrix
        hegst,               // Reduce a Hermitian-definite generalized eigenproblem to standard form
        hegv,                // Hermitian-definite generalized eigenproblem

        // SVD
        bdsvd,               // SVD for a bidiagonal matrix
        gesvd,               // SVD for a general matrix

        // BLAS
        trsm, // Temporarily exported until provided by cuBLASDx
    };
    inline constexpr auto potrf               = function::potrf;
    inline constexpr auto potrs               = function::potrs;
    inline constexpr auto posv                = function::posv;
    inline constexpr auto getrf_no_pivot      = function::getrf_no_pivot;
    inline constexpr auto getrs_no_pivot      = function::getrs_no_pivot;
    inline constexpr auto gesv_no_pivot       = function::gesv_no_pivot;
    inline constexpr auto getrf_partial_pivot = function::getrf_partial_pivot;
    inline constexpr auto getrs_partial_pivot = function::getrs_partial_pivot;
    inline constexpr auto gesv_partial_pivot  = function::gesv_partial_pivot;
    inline constexpr auto gtsv_no_pivot       = function::gtsv_no_pivot;
    inline constexpr auto modified_lu         = function::modified_lu;
    inline constexpr auto geqrf               = function::geqrf;
    inline constexpr auto unmqr               = function::unmqr;
    inline constexpr auto ungqr               = function::ungqr;
    inline constexpr auto gelqf               = function::gelqf;
    inline constexpr auto unmlq               = function::unmlq;
    inline constexpr auto unglq               = function::unglq;
    inline constexpr auto gels                = function::gels;
    inline constexpr auto htev                = function::htev;
    inline constexpr auto heev                = function::heev;
    inline constexpr auto hegst               = function::hegst;
    inline constexpr auto hegv                = function::hegv;
    inline constexpr auto bdsvd               = function::bdsvd;
    inline constexpr auto gesvd               = function::gesvd;
    inline constexpr auto trsm                = function::trsm;

    enum class job
    {
        no_vectors,        // Don't compute any vectors
        all_vectors,       // Compute all vectors in a separate array
        some_vectors,      // Compute min(m, n) vectors in a separate array
        multiply_vectors,  // Compute all vectors and multiply them to the existing array content
        overwrite_vectors, // Compute min(m, n) vectors and overwrite the A matrix
    };

    enum class side
    {
        left,
        right,
    };
    inline constexpr auto left  = side::left;
    inline constexpr auto right = side::right;

    enum class transpose
    {
        non_transposed,
        transposed,
        conj_transposed,
    };
    inline constexpr auto non_trans  = transpose::non_transposed;
    inline constexpr auto trans      = transpose::transposed;
    inline constexpr auto conj_trans = transpose::conj_transposed;
} // namespace cusolverdx

#endif // CUSOLVERDX_OPERATORS_ENUMS_HPP
