// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_ACCUMULATOR_HPP
#define CUBLASDX_DETAIL_BLAS_ACCUMULATOR_HPP

#include "cublasdx/detail/accumulator/rmem_accumulator.hpp"
#include "cublasdx/detail/accumulator/tmem_accumulator.hpp"

namespace cublasdx {
    namespace detail {

        template<class TiledMMA,
                 instruction_type InstructionKind,
                 class LoadInstruction,
                 class StoreInstruction,
                 class ShapeMN,
                 class InputTypeC,
                 class Alignment,
                 class HasStaticBlockDim,
                 class BlockSize,
                 class AccumulatorSM = cute::Int<1000>>
        using blas_accumulator = cute::conditional_t<
            InstructionKind == instruction_type::utcmma,
            tmem_accumulator<TiledMMA, InstructionKind, LoadInstruction, StoreInstruction,
                             ShapeMN, InputTypeC, Alignment, HasStaticBlockDim, BlockSize, AccumulatorSM>,
            rmem_accumulator<TiledMMA, InstructionKind, LoadInstruction, StoreInstruction,
                             ShapeMN, InputTypeC, Alignment, HasStaticBlockDim, BlockSize>>;

        template<class Partitioner>
        struct is_valid_blas_accumulator_impl: cute::false_type {};

        template<class... Args,
                 instruction_type InstructionKind,
                 class LoadInstruction,
                 class StoreInstruction,
                 class ShapeMN,
                 class InputTypeC,
                 class Alignment,
                 class HasStaticBlockDim,
                 class BlockSize>
        struct is_valid_blas_accumulator_impl<rmem_accumulator<cute::TiledMMA<Args...>,
                                                                InstructionKind,
                                                                LoadInstruction,
                                                                StoreInstruction,
                                                                ShapeMN,
                                                                InputTypeC,
                                                                Alignment,
                                                                HasStaticBlockDim,
                                                                BlockSize>>: cute::true_type {};

        template<class... Args,
                 instruction_type InstructionKind,
                 class LoadInstruction,
                 class StoreInstruction,
                 class ShapeMN,
                 class InputTypeC,
                 class Alignment,
                 class HasStaticBlockDim,
                 class BlockSize,
                 class AccumulatorSM>
        struct is_valid_blas_accumulator_impl<tmem_accumulator<cute::TiledMMA<Args...>,
                                                                InstructionKind,
                                                                LoadInstruction,
                                                                StoreInstruction,
                                                                ShapeMN,
                                                                InputTypeC,
                                                                Alignment,
                                                                HasStaticBlockDim,
                                                                BlockSize,
                                                                AccumulatorSM>>: cute::true_type {};

        template<class BlasAccumulator>
        struct is_valid_blas_accumulator: is_valid_blas_accumulator_impl<cute::remove_cvref_t<BlasAccumulator>> {};

        template<class BlasAccumulator>
        inline constexpr bool is_valid_blas_accumulator_v = is_valid_blas_accumulator<BlasAccumulator>::value;

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_BLAS_ACCUMULATOR_HPP
