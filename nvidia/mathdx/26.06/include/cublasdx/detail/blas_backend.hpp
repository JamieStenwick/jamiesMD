// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_GEMM_BACKEND_HPP
#define CUBLASDX_DATABASE_GEMM_BACKEND_HPP

#include "cublasdx/detail/cute/cute_tensor.hpp"
#include "cublasdx/database/gemm/database_getter.hpp"
#include "cublasdx/database/gemm/suggested/suggested_layouts.hpp"
#include "cublasdx/detail/blas_accumulator.hpp"

namespace cublasdx::detail {

    inline constexpr auto identity_swizzle = cute::Swizzle<0, 4, 3>{};

    template<class Engine>
    struct retrieve_pointer_swizzle {
        CUBLASDX_DEVICE constexpr auto operator()() {
            return identity_swizzle;
        }
    };

    template<class SwizzleFn, class Iterator>
    struct retrieve_pointer_swizzle<cute::swizzle_ptr<SwizzleFn,Iterator>> {
        CUBLASDX_DEVICE constexpr auto operator()() {
            return SwizzleFn{};
        }
    };

    template<class Engine, class Layout>
    CUBLASDX_DEVICE constexpr auto get_pointer_swizzle(cute::Tensor<Engine, Layout>) {
        return retrieve_pointer_swizzle<Engine>()();
    }

    template<class Tensor>
    struct get_layout {
        using type = decltype(Tensor().layout());
    };

    template<>
    struct get_layout<void> {
        using type = void;
    };

    template<class Tensor>
    using get_layout_t = typename get_layout<Tensor>::type;

    template<class BlockSize>
    CUBLASDX_DEVICE static constexpr unsigned get_threads() {
        unsigned ret = 128;
        if constexpr (not cute::is_void_v<BlockSize>) {
            ret = BlockSize::flat_size;
        }
        return ret;
    }

    template<class TiledMMA>
    CUBLASDX_DEVICE static constexpr unsigned get_mma_threads() {
        unsigned ret = 0;
        if constexpr (not cute::is_void_v<TiledMMA>) {
            ret = cute::size(TiledMMA {});
        }
        return ret;
    }

    template<class BlockSize>
    CUBLASDX_DEVICE static constexpr unsigned get_block_rank() {
        unsigned ret = 1;
        if constexpr (not cute::is_void_v<BlockSize>) {
            ret = BlockSize::rank;
        }
        return ret;
    }

    template<int BlockRank>
    CUBLASDX_DEVICE static unsigned int get_thread_idx() {
        constexpr int block_rank = BlockRank;
        if constexpr (block_rank == 3) {
            return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        } else if constexpr (block_rank == 2) {
            return threadIdx.x + threadIdx.y * blockDim.x;
        } else {
            return threadIdx.x;
        }
    }

    template<class BlockSize>
    CUBLASDX_DEVICE static unsigned int get_thread_idx() {
        constexpr int block_rank = get_block_rank<BlockSize>();
        return get_thread_idx<block_rank>();
    }

    template<typename Iter, class Layout>
    struct make_reference_swizzled_layout {
        using type = decltype(cute::make_tensor(cute::make_smem_ptr<Iter>(nullptr), Layout {}));
    };

    template<typename Iter>
    struct make_reference_swizzled_layout<Iter, void> {
        using type = void;
    };

    template<typename Iter, class Layout>
    using make_reference_swizzled_layout_t = typename make_reference_swizzled_layout<Iter, Layout>::type;

    template<bool UseSuggested,
             class DefaultAccumulator,
             class TiledMMA,
             instruction_type InstructionKind,
             class LoadInstruction,
             class StoreInstruction,
             class ShapeMN,
             class InputTypeC,
             class Alignment,
             class HasStaticBlockDim,
             class BlockSize,
             class AccumulatorSM>
    struct suggested_accumulator_type {
        using type = DefaultAccumulator;
    };

    template<class DefaultAccumulator,
             class TiledMMA,
             instruction_type InstructionKind,
             class LoadInstruction,
             class StoreInstruction,
             class ShapeMN,
             class InputTypeC,
             class Alignment,
             class HasStaticBlockDim,
             class BlockSize,
             class AccumulatorSM>
    struct suggested_accumulator_type<true,
                                      DefaultAccumulator,
                                      TiledMMA,
                                      InstructionKind,
                                      LoadInstruction,
                                      StoreInstruction,
                                      ShapeMN,
                                      InputTypeC,
                                      Alignment,
                                      HasStaticBlockDim,
                                      BlockSize,
                                      AccumulatorSM> {
        using type = blas_accumulator<TiledMMA,
                                      InstructionKind,
                                      LoadInstruction,
                                      StoreInstruction,
                                      ShapeMN,
                                      InputTypeC,
                                      Alignment,
                                      HasStaticBlockDim,
                                      BlockSize,
                                      AccumulatorSM>;
    };

    template<bool UseSuggested,
             class DefaultAccumulator,
             class TiledMMA,
             instruction_type InstructionKind,
             class LoadInstruction,
             class StoreInstruction,
             class ShapeMN,
             class InputTypeC,
             class Alignment,
             class HasStaticBlockDim,
             class BlockSize,
             class AccumulatorSM>
    using suggested_accumulator_type_t = typename suggested_accumulator_type<UseSuggested,
                                                                            DefaultAccumulator,
                                                                            TiledMMA,
                                                                            InstructionKind,
                                                                            LoadInstruction,
                                                                            StoreInstruction,
                                                                            ShapeMN,
                                                                            InputTypeC,
                                                                            Alignment,
                                                                            HasStaticBlockDim,
                                                                            BlockSize,
                                                                            AccumulatorSM>::type;

    namespace cute_backend {
        template<typename TypeA,
                 typename TypeB,
                 typename TypeC,
                 typename InputA,
                 typename InputB,
                 typename InputC,
                 typename Alignment,
                 int SizeM,
                 int SizeN,
                 int SizeK,
                 typename Arrangement,
                 typename TransposeMode,
                 typename SM,
                 sm_modifier SMModifier,
                 class HasStaticBlockDim,
                 class HasStreaming,
                 class HasWithPipeline,
                 typename BlockSize, // void if empty
                 typename OverloadedTileOperator>
        struct execution {
        private:
            static constexpr bool is_tile_overloaded = OverloadedTileOperator::valid;

            static constexpr int blocksize_threads = cute::size(
                get_database_threads_t<TypeA, TypeB, TypeC, SizeM, SizeN, SizeK, SM, SMModifier, BlockSize> {});
            static constexpr int overloaded_threads = cute::size(get_database_threads_t<TypeA,
                                                                                        TypeB,
                                                                                        TypeC,
                                                                                        SizeM,
                                                                                        SizeN,
                                                                                        SizeK,
                                                                                        SM,
                                                                                        SMModifier,
                                                                                        BlockSize,
                                                                                        OverloadedTileOperator> {});
            static_assert(blocksize_threads == overloaded_threads);

            // Necessary only for pointer API
            using blas_transpose_mode       = TransposeMode;
            static constexpr auto tr_mode_a = blas_transpose_mode::a_transpose_mode;
            static constexpr auto tr_mode_b = blas_transpose_mode::b_transpose_mode;
            static constexpr auto tr_mode_c = transpose_mode::non_transposed;

            using blas_blockdim = BlockSize;

            // Necessary only for pointer API
            using blas_arrangement      = Arrangement;
            static constexpr auto arr_a = blas_arrangement::a;
            static constexpr auto arr_b = blas_arrangement::b;
            static constexpr auto arr_c = blas_arrangement::c;

            using blas_alignment          = Alignment;
            static constexpr auto align_a = blas_alignment::a;
            static constexpr auto align_b = blas_alignment::b;
            static constexpr auto align_c = blas_alignment::c;

            // These are "safe" because they can always be passed to cute::copy, not the case
            // with e.g. LDSM
            using safe_a_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_a * 8>;
            using safe_b_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_b * 8>;
            using safe_c_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_c * 8>;

            // Necessary for both APIs
            static constexpr unsigned int m = SizeM;
            static constexpr unsigned int n = SizeN;
            static constexpr unsigned int k = SizeK;

            static constexpr int  block_size = get_threads<BlockSize>();
            static constexpr bool is_blockdim_static_v =
                (HasStaticBlockDim {}) and (block_size % 32 == 0);

            static constexpr auto is_blockdim_static = cute::constant<bool, is_blockdim_static_v> {};
            using is_blockdim_static_t               = cute::remove_cvref_t<decltype(is_blockdim_static)>;

            using swizzled_meta_info =
                cublasdx::detail::layout_database::optimal_config<block_size,
                                                                  SM::value,
                                                                  SMModifier,
                                                                  HasStreaming,
                                                                  HasWithPipeline,
                                                                  TypeA,
                                                                  arr_a == arrangement::col_major,
                                                                  align_a,
                                                                  TypeB,
                                                                  arr_b == arrangement::col_major,
                                                                  align_b,
                                                                  TypeC,
                                                                  arr_c == arrangement::col_major,
                                                                  align_c,
                                                                  m,
                                                                  n,
                                                                  k>;

            using swizzled_config          = typename swizzled_meta_info::tiled_mma;
            using swizzled_a_layout        = typename swizzled_meta_info::a_layout;
            using swizzled_a_tensor_t      = make_reference_swizzled_layout_t<TypeA, swizzled_a_layout>;
            using swizzled_b_layout        = typename swizzled_meta_info::b_layout;
            using swizzled_b_tensor_t      = make_reference_swizzled_layout_t<TypeB, swizzled_b_layout>;
            using swizzled_c_layout        = typename swizzled_meta_info::c_layout;
            using swizzled_a_copy_op       = typename swizzled_meta_info::a_copy_op;
            using swizzled_b_copy_op       = typename swizzled_meta_info::b_copy_op;
            using swizzled_c_copy_load_op  = typename swizzled_meta_info::c_copy_load_op;
            using swizzled_c_copy_store_op = typename swizzled_meta_info::c_copy_store_op;

            static constexpr bool is_swizzled_config_viable = not is_tile_overloaded and swizzled_meta_info::valid;

            // This is necessary for a case where swizzled config is available but the user
            // does not utilize suggested layout and db_entry has higher number of threads
            // than 128
            using db_config_meta = get_database_config<TypeA,
                                                       TypeB,
                                                       TypeC,
                                                       m,
                                                       n,
                                                       k,
                                                       arr_a,
                                                       arr_b,
                                                       arr_c,
                                                       align_a,
                                                       align_b,
                                                       align_c,
                                                       SM,
                                                       BlockSize,
                                                       OverloadedTileOperator>;

            using db_config          = typename db_config_meta::type;
            using db_a_copy_op       = typename db_config_meta::a_copy_op;
            using db_b_copy_op       = typename db_config_meta::b_copy_op;
            using db_c_copy_load_op  = typename db_config_meta::c_copy_load_op;
            using db_c_copy_store_op = typename db_config_meta::c_copy_store_op;

            using default_config          = db_config;
            using default_a_copy_op       = db_a_copy_op;
            using default_b_copy_op       = db_b_copy_op;
            using default_c_copy_load_op  = db_c_copy_load_op;
            using default_c_copy_store_op = db_c_copy_store_op;

            static constexpr int default_atom_size_threads = cute::size(typename default_config::AtomThrID {});
            static constexpr int default_atom_size_mnk     = cute::size(typename default_config::AtomShape_MNK {});

            // Works for FFMA and FFMA2
            static constexpr bool default_atom_is_fma =
                (default_atom_size_threads == 1) and (default_atom_size_mnk <= 2);
            static constexpr bool default_atom_is_supermma    = (default_atom_size_threads == 32);

            static_assert(default_atom_is_fma or default_atom_is_supermma);


            using suggested_threads_config =
                cute::conditional_t<is_swizzled_config_viable, swizzled_config, default_config>;

        public:
            static constexpr instruction_type swizzled_instruction_kind = swizzled_meta_info::instruction_kind;
            static constexpr instruction_type default_instruction_kind =
                default_atom_is_fma ? instruction_type::fma : instruction_type::supermma;
            static constexpr unsigned int suggested_threads = cute::size(typename suggested_threads_config::ThrLayoutVMNK{});

            using shape_mn_t = cute::Shape<cute::Int<SizeM>, cute::Int<SizeN>>;
            using default_accumulator_t = blas_accumulator<default_config,
                                                           default_instruction_kind,
                                                           default_c_copy_load_op,
                                                           default_c_copy_store_op,
                                                           shape_mn_t,
                                                           InputC,
                                                           cute::Int<align_c>,
                                                           is_blockdim_static_t,
                                                           cute::Int<blocksize_threads>,
                                                           cute::Int<SM::value>>;
            using suggested_accumulator_t = suggested_accumulator_type_t<is_swizzled_config_viable,
                                                                         default_accumulator_t,
                                                                         swizzled_config,
                                                                         swizzled_instruction_kind,
                                                                         swizzled_c_copy_load_op,
                                                                         swizzled_c_copy_store_op,
                                                                         shape_mn_t,
                                                                         InputC,
                                                                         cute::Int<align_c>,
                                                                         is_blockdim_static_t,
                                                                         cute::Int<blocksize_threads>,
                                                                         cute::Int<SM::value>>;

            CUBLASDX_DEVICE static default_accumulator_t get_accumulator() {
                const auto thread_idx = cublasdx::detail::get_thread_idx<BlockSize>();
                return default_accumulator_t(thread_idx);
            }

            CUBLASDX_DEVICE static suggested_accumulator_t suggest_accumulator() {
                const auto thread_idx = cublasdx::detail::get_thread_idx<BlockSize>();
                if constexpr (is_swizzled_config_viable) {
                    return suggested_accumulator_t(thread_idx);
                } else {
                    return get_accumulator();
                }
            }

            template<typename ATensor, typename BTensor>
            CUBLASDX_DEVICE static auto choose_accumulator(ATensor const& atensor, BTensor const& btensor) {
                using a_tensor_t = cute::remove_cvref_t<ATensor>;
                using b_tensor_t = cute::remove_cvref_t<BTensor>;

                constexpr auto swizzle_a_tensor_pointer = [=]() constexpr {
                    if constexpr(is_swizzled_config_viable) {
                        return get_pointer_swizzle(swizzled_a_tensor_t{});
                    } else {
                        return identity_swizzle;
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                constexpr auto swizzle_b_tensor_pointer = [=]() constexpr {
                    if constexpr(is_swizzled_config_viable) {
                        return get_pointer_swizzle(swizzled_b_tensor_t{});
                    } else {
                        return identity_swizzle;
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                constexpr bool is_suggested_ab_pointer_swizzles =
                    decltype(get_pointer_swizzle(atensor)){} == swizzle_a_tensor_pointer and
                    decltype(get_pointer_swizzle(btensor)){} == swizzle_b_tensor_pointer;

                constexpr bool is_suggested_mma =
                    is_swizzled_config_viable and
                    cute::is_same_v<get_layout_t<a_tensor_t>, get_layout_t<swizzled_a_tensor_t>> and
                    cute::is_same_v<get_layout_t<b_tensor_t>, get_layout_t<swizzled_b_tensor_t>> and
                    is_suggested_ab_pointer_swizzles;

                if constexpr (is_suggested_mma) {
                    return suggest_accumulator();
                } else {
                    return get_accumulator();
                }

                CUTE_GCC_UNREACHABLE;
            }



            // C in registers API
            template<bool GatingTurnedOff = false,
                     typename TSA,
                     typename ALayout,
                     typename TSB,
                     typename BLayout,
                     typename BlasAccumulator,
                     typename ALoadOp = identity,
                     typename BLoadOp = identity,
                     __CUTE_REQUIRES(cute::is_smem_v<TSA>and cute::is_smem_v<TSB>and
                                                             is_valid_blas_accumulator<BlasAccumulator>::value)>
            CUBLASDX_DEVICE static void tensor_gemm(cute::Tensor<TSA, ALayout> const& smem_tensor_a,
                                                    cute::Tensor<TSB, BLayout> const& smem_tensor_b,
                                                    BlasAccumulator&                  accumulator,
                                                    const ALoadOp&                    a_load_op = identity {},
                                                    const BLoadOp&                    b_load_op = identity {}) {
                const auto thread_idx = cublasdx::detail::get_thread_idx<BlockSize>();

                using a_tensor_t = cute::remove_cvref_t<decltype(smem_tensor_a)>;
                using b_tensor_t = cute::remove_cvref_t<decltype(smem_tensor_b)>;

                constexpr bool is_suggested_ab_layouts =
                    cute::is_same_v<get_layout_t<a_tensor_t>, get_layout_t<swizzled_a_tensor_t>> and
                    cute::is_same_v<get_layout_t<b_tensor_t>, get_layout_t<swizzled_b_tensor_t>>;

                constexpr auto swizzle_a_tensor_pointer = [=]() constexpr {
                    if constexpr(is_swizzled_config_viable) {
                        return get_pointer_swizzle(swizzled_a_tensor_t{});
                    } else {
                        return identity_swizzle;
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                constexpr auto swizzle_b_tensor_pointer = [=]() constexpr {
                    if constexpr(is_swizzled_config_viable) {
                        return get_pointer_swizzle(swizzled_b_tensor_t{});
                    } else {
                        return identity_swizzle;
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                constexpr bool is_suggested_ab_pointer_swizzles =
                    decltype(get_pointer_swizzle(smem_tensor_a)){} == swizzle_a_tensor_pointer and
                    decltype(get_pointer_swizzle(smem_tensor_b)){} == swizzle_b_tensor_pointer;

                constexpr bool is_suggested_ab = is_suggested_ab_layouts and is_suggested_ab_pointer_swizzles;

                using suggested_acc_t = cute::remove_cvref_t<decltype(suggest_accumulator())>;
                using default_acc_t   = cute::remove_cvref_t<decltype(get_accumulator())>;

                constexpr bool is_suggested_fragment =
                    cute::is_same_v<cute::remove_cvref_t<BlasAccumulator>, suggested_acc_t>;
                constexpr bool is_suggested_same_as_default = cute::is_same_v<suggested_acc_t, default_acc_t>;
                constexpr bool are_identity_ops             = is_identity_op_v<ALoadOp> and is_identity_op_v<BLoadOp>;

                static_assert((not is_swizzled_config_viable or is_suggested_same_as_default) or
                              (not is_suggested_fragment or is_suggested_ab),
                              "Suggested accumulator can be used only with suggested layouts");

                constexpr bool is_suggested_mma = ((swizzled_instruction_kind == instruction_type::supermma or
                                                    swizzled_instruction_kind == instruction_type::fma) or
                                                   are_identity_ops) and
                                                  is_swizzled_config_viable and is_suggested_ab and
                                                  is_suggested_fragment;

                static_assert(cute::is_same_v<cute::remove_cvref_t<BlasAccumulator>, suggested_acc_t> or
                              cute::is_same_v<cute::remove_cvref_t<BlasAccumulator>, default_acc_t>);

                static_assert(
                    (is_suggested_mma and cute::is_same_v<cute::remove_cvref_t<BlasAccumulator>, suggested_acc_t>) or
                        (not is_suggested_mma and
                         cute::is_same_v<cute::remove_cvref_t<BlasAccumulator>, default_acc_t>),
                    "Incompatible C fragment type used");


                using chosen_mma_t      = cute::conditional_t<is_suggested_mma, swizzled_config, default_config>;
                using accumulator_mma_t = cute::remove_cvref_t<decltype(accumulator.tiled_mma)>;

                static_assert(cute::is_same_v<chosen_mma_t, accumulator_mma_t>);

                constexpr bool gating_turned_off = GatingTurnedOff;

                if constexpr ((not is_suggested_mma) or swizzled_instruction_kind == instruction_type::supermma or
                              swizzled_instruction_kind == instruction_type::fma) {

                    // Check if decoupled precision was used for A
                    constexpr int a_layout_alignment =
                        cute::is_static_v<ALayout> ? cute::gcd(align_a, cute::max_alignment(ALayout {}) * sizeof(TypeA))
                                                   : sizeof(TypeA);

                    constexpr bool is_type_copy_compatible_a = cute::is_static_v<ALayout> and
                                                               (a_layout_alignment == align_a) and
                                                               (sizeof(typename TSA::value_type) == sizeof(TypeA) and
                                                                alignof(typename TSA::value_type) == alignof(TypeA));

                    // Check if decoupled precision was used for B
                    constexpr int b_layout_alignment =
                        cute::is_static_v<BLayout> ? cute::gcd(align_b, cute::max_alignment(BLayout {}) * sizeof(TypeB))
                                                   : sizeof(TypeB);
                    constexpr bool is_type_copy_compatible_b = cute::is_static_v<BLayout> and
                                                               (b_layout_alignment == align_b) and
                                                               (sizeof(typename TSB::value_type) == sizeof(TypeB) and
                                                                alignof(typename TSB::value_type) == alignof(TypeB));

                    using a_fast_copy_op = cute::conditional_t<is_suggested_mma, swizzled_a_copy_op, default_a_copy_op>;
                    using a_vec_copy_op  = cute::AutoVectorizingCopyWithAssumedAlignment<a_layout_alignment * 8>;
                    auto a_copy_op = cute::conditional_t<is_type_copy_compatible_a, a_fast_copy_op, a_vec_copy_op> {};

                    using b_fast_copy_op = cute::conditional_t<is_suggested_mma, swizzled_b_copy_op, default_b_copy_op>;
                    using b_vec_copy_op  = cute::AutoVectorizingCopyWithAssumedAlignment<b_layout_alignment * 8>;
                    auto b_copy_op = cute::conditional_t<is_type_copy_compatible_b, b_fast_copy_op, b_vec_copy_op> {};

                    if (gating_turned_off or
                        (is_blockdim_static and blocksize_threads == cute::size(accumulator.tiled_mma)) or
                        (thread_idx < cute::size(accumulator.tiled_mma))) {
                        cute::cooperative_gemm(thread_idx,
                                               accumulator.tiled_mma,
                                               smem_tensor_a,
                                               transpose_first_two_modes_of_tensor(smem_tensor_b),
                                               accumulator.accumulator_storage,
                                               a_load_op,
                                               b_load_op,
                                               a_copy_op,
                                               b_copy_op);
                    }
                } else if constexpr (is_suggested_mma and swizzled_instruction_kind == instruction_type::gmma) {
                    asm volatile("fence.proxy.async.shared::cta; \n" ::: "memory");

                    cute::Tensor tCsA = accumulator.sliced_mma.partition_A(smem_tensor_a);
                    cute::Tensor tCsB = accumulator.sliced_mma.partition_B(transpose_first_two_modes_of_tensor(smem_tensor_b));

                    // Shared memory fragment creation
                    cute::Tensor tCrA = accumulator.sliced_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
                    cute::Tensor tCrB = accumulator.sliced_mma.make_fragment_B(tCsB);

                    cute::warpgroup_fence_operand(accumulator.accumulator_storage);
                    cute::warpgroup_arrive();

                    if (gating_turned_off or
                        (is_blockdim_static and blocksize_threads == cute::size(accumulator.tiled_mma)) or
                        (thread_idx < cute::size(accumulator.tiled_mma))) {
                        CUTE_UNROLL
                        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                            cute::gemm(accumulator.tiled_mma,
                                       tCrA(cublasdx::slice, cublasdx::slice, k_block),
                                       tCrB(cublasdx::slice, cublasdx::slice, k_block),
                                       accumulator.accumulator_storage);
                            // After first MMA iteration the accumulation mode can be switched to 
                            // C += A * B
                            // from 
                            // C = A * B
                            accumulator.turn_on_accumulation();
                        }
                    }

                    cute::warpgroup_commit_batch();
                    cute::warpgroup_wait<0>();
                    cute::warpgroup_fence_operand(accumulator.accumulator_storage);
                } else if constexpr (is_suggested_mma and swizzled_instruction_kind == instruction_type::utcmma) {
                    // Thread partition
                    cute::Tensor tCsA = accumulator.sliced_mma.partition_A(smem_tensor_a);
                    cute::Tensor tCsB = accumulator.sliced_mma.partition_B(transpose_first_two_modes_of_tensor(smem_tensor_b));

                    // Shared memory fragment creation
                    cute::Tensor tCrA = accumulator.sliced_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
                    cute::Tensor tCrB = accumulator.sliced_mma.make_fragment_B(tCsB);

                    auto acc_for_gemm = accumulator.compute_accumulator();

                    CUTE_UNROLL
                    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                        cute::gemm(accumulator.tiled_mma,
                                   tCrA(cublasdx::slice, cublasdx::slice, k_block),
                                   tCrB(cublasdx::slice, cublasdx::slice, k_block),
                                   acc_for_gemm);
                        accumulator.turn_on_accumulation();
                    }
                } else {
                    static_assert(not is_suggested_mma or (swizzled_instruction_kind == instruction_type::gmma or
                                                           swizzled_instruction_kind == instruction_type::utcmma),
                                  "Invalid path");
                }
            }
        };
    } // namespace cute_backend
} // namespace cublasdx::detail

#endif // CUBLASDX_DATABASE_GEMM_BACKEND_HPP
