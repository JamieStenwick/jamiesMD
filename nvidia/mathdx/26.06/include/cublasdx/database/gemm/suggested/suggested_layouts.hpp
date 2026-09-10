// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_SUGGESTED_LAYOUTS_HPP
#define CUBLASDX_SUGGESTED_LAYOUTS_HPP

#include "cublasdx/detail/cute/cute_tensor.hpp"
#include "cublasdx/database/gemm/mma_atoms.hpp"
#include "cublasdx/database/gemm/suggested/suggested_instructions.hpp"

namespace cublasdx {
    namespace detail {
        namespace layout_database {
            using cublasdx::detail::cute_backend::transpose_first_two_modes_of_layout;

            enum class config_type
            {
                variable_atom_tile,
                fixed_atom_tile
            };

            enum class swizzle_type {
                position_independent,
                position_dependent
            };

            enum class execution_type {
                cta_tile,
                device_pipeline
            };

            template<class LayoutShape, bool IsKMajor, class ElemType, swizzle_type SwizzleType>
            CUBLASDX_HOST_DEVICE auto constexpr get_ldsm_swizzled_layout() {
                // Setup constants
                constexpr unsigned cache_line_bytes        = 128;
                constexpr unsigned max_vectorization_bytes = 16;

                // Process inputs
                constexpr unsigned m = cute::tuple_element_t<0, LayoutShape>::value;
                constexpr unsigned k = cute::tuple_element_t<1, LayoutShape>::value;

                // 1. Generate major dimension of subtile shared-memory shape
                constexpr unsigned num_elems_per_cache_line = cache_line_bytes / sizeof(ElemType);
                constexpr unsigned major_layout_shape       = IsKMajor ? k : m;
                constexpr unsigned minor_layout_shape       = IsKMajor ? m : k;

                // GCD is a power of 2 at least equal to warp tiled atom major length
                constexpr unsigned major_shared_memory_atom_length =
                    cute::gcd(num_elems_per_cache_line, major_layout_shape);

                // 2. Generate Swizzle --> MBase
                constexpr unsigned vectorization_level = max_vectorization_bytes / sizeof(ElemType);
                constexpr unsigned swizzle_mbase       = log2(vectorization_level);

                // 3. Generate Swizzle --> SShift
                constexpr unsigned swizzle_sshift = log2(num_elems_per_cache_line) - swizzle_mbase;

                // 4. Generate Swizzle --> BBits
                constexpr unsigned vec_elems_per_cache_line =
                    cache_line_bytes / (vectorization_level * sizeof(ElemType));
                constexpr unsigned rows_per_cache_line =
                    cache_line_bytes / (major_shared_memory_atom_length * sizeof(ElemType));
                constexpr unsigned swizzle_bbits = log2(vec_elems_per_cache_line / rows_per_cache_line);

                // 5. Generate minor dimension of subtile shared-memory shape
                constexpr unsigned necessary_swizzled_dim =
                    1 << (swizzle_mbase + swizzle_sshift + swizzle_bbits - log2(major_shared_memory_atom_length));

                // GCD is a power of 2 at least equal to warp tiled atom minor length
                constexpr unsigned minor_shared_memory_atom_length =
                    cute::gcd(minor_layout_shape, necessary_swizzled_dim);

                // 6. Depending on layout, choose major and minor side
                constexpr unsigned layout_atom_shape_m =
                    IsKMajor ? minor_shared_memory_atom_length : major_shared_memory_atom_length;
                constexpr unsigned layout_atom_shape_k =
                    IsKMajor ? major_shared_memory_atom_length : minor_shared_memory_atom_length;

                static_assert(m % layout_atom_shape_m == 0, "Suggested layout atom divisibility condition failed");
                static_assert(k % layout_atom_shape_k == 0, "Suggested layout atom divisibility condition failed");

                // Signed conversion
                constexpr int int_bbits  = static_cast<int>(swizzle_bbits);
                constexpr int int_mbase  = static_cast<int>(swizzle_mbase);
                constexpr int int_sshift = static_cast<int>(swizzle_sshift);

                static_assert(int_bbits <= 5 and int_mbase <= 4 and int_sshift <= 7);

                if constexpr(int_bbits == 0) {
                    constexpr auto smem_atom = 
                        cute::make_layout(
                            cute::make_shape(cute::Int<layout_atom_shape_m> {}, cute::Int<layout_atom_shape_k> {}),
                            cute::conditional_return<IsKMajor>(cute::LayoutRight {}, cute::LayoutLeft {}));
                    return remove_zeros_from_layout(cute::tile_to_shape(smem_atom, LayoutShape {}));
                } else if constexpr(SwizzleType == swizzle_type::position_dependent) {
                    constexpr auto smem_atom = cute::composition(
                        cute::downcast<sizeof(ElemType)>(cute::Swizzle<int_bbits, int_mbase, int_sshift>()),
                        cute::smem_ptr_flag_bits<cute::sizeof_bits_v<ElemType>>(),
                        cute::make_layout(
                            cute::make_shape(cute::Int<layout_atom_shape_m> {}, cute::Int<layout_atom_shape_k> {}),
                            cute::conditional_return<IsKMajor>(cute::LayoutRight {}, cute::LayoutLeft {})));
		            return remove_zeros_from_layout(cute::tile_to_shape(smem_atom, LayoutShape {}));
                } else {
                    constexpr auto smem_atom = cute::composition(
                        cute::Swizzle<int_bbits, int_mbase, int_sshift>(),
                        cute::C<0U>{},
                        cute::make_layout(
                            cute::make_shape(cute::Int<layout_atom_shape_m> {}, cute::Int<layout_atom_shape_k> {}),
                            cute::conditional_return<IsKMajor>(cute::LayoutRight {}, cute::LayoutLeft {})));
                    return remove_zeros_from_layout(cute::tile_to_shape(smem_atom, LayoutShape {}));
                }

                CUTE_GCC_UNREACHABLE;
            }

            template<class ElemType, class = void>
            struct precision_helper {
                using type = ElemType;
            };

            template<class ElemType>
            struct precision_helper<ElemType, cute::enable_if_t<has_complex_interface_v<ElemType>>> {
                using type = typename ElemType::value_type;
            };

            template<class LayoutShape, int MPermutation, bool IsKMajor, class ElemType>
            CUBLASDX_HOST_DEVICE auto constexpr get_lds_swizzled_layout() {
                // Setup constants
                constexpr unsigned cache_line_bytes = 128;
                constexpr unsigned single_bank_size = sizeof(float);
                // Element is always 32 bit, apart from double for which it's 64 bit
                using precision                        = typename precision_helper<ElemType>::type;
                constexpr unsigned bytes_per_precision = cute::max(sizeof(precision), sizeof(float));
                constexpr unsigned bytes_per_load = (has_complex_interface<ElemType>() ? 2 : 1) * bytes_per_precision;

                // Process inputs
                constexpr unsigned m = cute::tuple_element_t<0, LayoutShape>::value;
                constexpr unsigned k = cute::tuple_element_t<1, LayoutShape>::value;

                // 1. Generate major dimension of subtile shared-memory shape
                constexpr unsigned num_elems_per_cache_line = cache_line_bytes / sizeof(ElemType);
                constexpr unsigned major_layout_shape       = IsKMajor ? k : m;
                constexpr unsigned minor_layout_shape       = IsKMajor ? m : k;

                // GCD is a power of 2 at least equal to warp multiplied major atom length
                constexpr unsigned major_shared_memory_atom_length =
                    cute::gcd(num_elems_per_cache_line, major_layout_shape);

                constexpr unsigned vectorization_level = IsKMajor ? (bytes_per_load / sizeof(ElemType)) : MPermutation;

                // 2. Generate Swizzle --> BBits
                constexpr unsigned transaction_size_bytes = vectorization_level * sizeof(ElemType);
                constexpr unsigned transaction_threads = 32 / cute::max(1, transaction_size_bytes / single_bank_size);

                constexpr unsigned threads_per_mma_row = cute::tuple_element_t<1, warp_thread_layout_supermma> {};
                constexpr unsigned conflicting_threads =
                    IsKMajor ? (transaction_threads / threads_per_mma_row) : (threads_per_mma_row);
                constexpr unsigned rows_per_cache_line = num_elems_per_cache_line / major_shared_memory_atom_length;
                constexpr unsigned swizzle_bbits       = log2(cute::max(1, conflicting_threads / rows_per_cache_line));

                // 3. Generate Swizzle --> MBase, this time we check to generate possible permutations
                constexpr unsigned num_conflicting_rows =
                    IsKMajor ? threads_per_mma_row : (transaction_threads / threads_per_mma_row);
                // The trivial case is when all threads are hitting the same row, it gets worse when data is col-major
                // and consecutive rows of 4 threads will try to (when trivially swizzled) access same banks
                constexpr unsigned swizzle_mbase = log2(vectorization_level * num_conflicting_rows);

                // 4. Generate Swizzle --> SShift
                constexpr unsigned swizzle_sshift = log2(num_elems_per_cache_line) - swizzle_mbase;

                // 5. Generate minor dimension of subtile shared-memory shape
                constexpr unsigned necessary_swizzled_dim =
                    1 << (swizzle_mbase + swizzle_sshift + swizzle_bbits - log2(major_shared_memory_atom_length));
                // GCD is a power of 2 at least equal to warp multiplied minor atom length
                constexpr unsigned minor_shared_memory_atom_length =
                    cute::gcd(minor_layout_shape, necessary_swizzled_dim);

                // 6. Depending on layout, choose major and minor side
                constexpr unsigned layout_atom_shape_m =
                    IsKMajor ? minor_shared_memory_atom_length : major_shared_memory_atom_length;
                constexpr unsigned layout_atom_shape_k =
                    IsKMajor ? major_shared_memory_atom_length : minor_shared_memory_atom_length;

                static_assert(m % layout_atom_shape_m == 0, "Suggested layout atom divisibility condition failed");
                static_assert(k % layout_atom_shape_k == 0, "Suggested layout atom divisibility condition failed");

                // Signed conversion
                constexpr int int_bbits  = static_cast<int>(swizzle_bbits);
                constexpr int int_mbase  = static_cast<int>(swizzle_mbase);
                constexpr int int_sshift = static_cast<int>(swizzle_sshift);

                // Sanity check - there is never a reason to have swizzles outside these values:
                // bbits <= 5, because maximally 32 elements (2^5) will need to be permuted
                // mbase <= 5, because max vectorization level is 16 (2^4 for 8bit elems),
                // and max 1 extra level needs to be added for 2 consecutive rows (4 + 1 == 5)
                // sshift <= 7, because maximally 128 elements are in a cache-line (128 == 2^7)
                static_assert(int_bbits <= 5 and int_mbase <= 5 and int_sshift <= 7);

                // For now always emit PISL, PDSL possible but without clear gains
                if constexpr(int_bbits == 0) {
                    constexpr auto smem_atom = 
                        cute::make_layout(
                            cute::make_shape(cute::Int<layout_atom_shape_m> {}, cute::Int<layout_atom_shape_k> {}),
                            cute::conditional_return<IsKMajor>(cute::LayoutRight {}, cute::LayoutLeft {}));
                    return remove_zeros_from_layout(cute::tile_to_shape(smem_atom, LayoutShape {}));
                } else {
                    constexpr auto smem_atom = cute::composition(
                        cute::Swizzle<int_bbits, int_mbase, int_sshift>(),
                        cute::C<0U>{},
                        cute::make_layout(
                            cute::make_shape(cute::Int<layout_atom_shape_m> {}, cute::Int<layout_atom_shape_k> {}),
                            cute::conditional_return<IsKMajor>(cute::LayoutRight {}, cute::LayoutLeft {})));

                    return remove_zeros_from_layout(cute::tile_to_shape(smem_atom, LayoutShape {}));
                }

                CUTE_GCC_UNREACHABLE;
            }

            template<class LDSMInstruction,
                     class STSMInstruction,
                     class AtomShape,
                     class LayoutShape,
                     bool IsKMajor,
                     class ElemType,
                     swizzle_type SwizzleType>
            CUBLASDX_HOST_DEVICE auto constexpr get_ldsm_swizzled_config() {
                // Process input --> Using A Matrix naming, B must be transposed to use this
                constexpr int warp_m = cute::tuple_element_t<0, AtomShape>::value;
                constexpr int atom_m = cute::tuple_element_t<1, AtomShape>::value;
                constexpr int atom_k = cute::tuple_element_t<2, AtomShape>::value;

                // Get Layout optimized for accessing rows and columns at the same time
                constexpr auto layout = get_ldsm_swizzled_layout<LayoutShape, IsKMajor, ElemType, SwizzleType>();

                // Copy instruction --> Attempt LDSMx4
                // Value Modifier
                constexpr int thread_adjusted_atom_length = warp_m * atom_m;
                constexpr int ldsm_bytes = cute::size(typename cute::Copy_Traits<LDSMInstruction>::RefLayout {}) / 8;

                if constexpr (not cute::is_same_v<STSMInstruction, void>) {
                    constexpr int stsm_bytes =
                        cute::size(typename cute::Copy_Traits<STSMInstruction>::RefLayout {}) / 8;
                    static_assert(ldsm_bytes == stsm_bytes);
                }

                constexpr auto m_value_tile = cute::Int<cute::ceil_div(ldsm_bytes, atom_m * atom_k * sizeof(ElemType)) *
                                                        thread_adjusted_atom_length> {};

                using load_op              = LDSMInstruction;
                using autovectorizing_copy = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
                using store_op =
                    cute::conditional_t<cute::is_same_v<STSMInstruction, void>, autovectorizing_copy, STSMInstruction>;

                return cute::make_tuple(layout, m_value_tile, load_op {}, store_op {});
            }

            template<class AtomShape,
                     class LayoutShape,
                     bool IsKMajor,
                     class ElemType,
                     int         Alignment,
                     config_type Config>
            CUBLASDX_HOST_DEVICE auto constexpr get_lds_swizzled_config() {
                constexpr unsigned max_vectorization_bytes = Alignment;
                constexpr bool     proper_alignment =
                    (Alignment >= sizeof(ElemType)) and (Alignment % sizeof(ElemType) == 0);
                // Process input --> Using A Matrix naming, B must be transposed to use this
                constexpr unsigned warp_m = cute::tuple_element_t<0, AtomShape>::value;
                constexpr unsigned atom_m = cute::tuple_element_t<1, AtomShape>::value;
                constexpr unsigned gemm_m = cute::tuple_element_t<0, LayoutShape>::value;

                constexpr bool     is_fixed_config = Config == config_type::fixed_atom_tile;
                constexpr unsigned m_permutation =
                    (IsKMajor or is_fixed_config or not proper_alignment)
                        ? 1
                        : cute::gcd(gemm_m / (warp_m * atom_m), max_vectorization_bytes / sizeof(ElemType));

                // Get Layout optimized for accessing rows and columns at the same time
                constexpr auto layout = get_lds_swizzled_layout<LayoutShape, m_permutation, IsKMajor, ElemType>();

                // Copy instruction --> Attempt LDSMx4
                // Value Modifier
                constexpr int  thread_adjusted_atom_length = warp_m * atom_m;
                constexpr auto m_value_tile =
                    cute::Layout<cute::Shape<cute::Int<thread_adjusted_atom_length>, cute::Int<m_permutation>>,
                                 cute::Stride<cute::Int<m_permutation>, cute::_1>> {};
                constexpr auto copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<Alignment * 8> {};

                return cute::make_tuple(layout, m_value_tile, copy_op, copy_op);
            }

            template<int SM,
                     class AtomShape,
                     class LayoutShape,
                     bool IsKMajor,
                     class ElemType,
                     int   Alignment,
                     execution_type  ExecutionType,
                     config_type Config = config_type::variable_atom_tile>
            CUBLASDX_HOST_DEVICE auto constexpr get_swizzled_config() {
                constexpr int ldsm_alignment_requirement = 16;
                // Get LDST instructions
                using limiting_shape = cute::conditional_t<Config == config_type::variable_atom_tile,
                                                           LayoutShape,
                                                           decltype(cute::select<1, 2>(AtomShape {}))>;

                using ldsm_instruction_t =
                    get_best_ldsm_instruction_t<SM, limiting_shape, AtomShape, ElemType, IsKMajor>;
                if constexpr (cute::is_void_v<ldsm_instruction_t> or Alignment != ldsm_alignment_requirement) {
                    /* Attempt LDS.128 or LDS.64 swizzle vectorization */
                    return get_lds_swizzled_config<AtomShape, LayoutShape, IsKMajor, ElemType, Alignment, Config>();
                } else {
                    using stsm_instruction_t =
                        get_best_stsm_instruction_t<SM, LayoutShape, AtomShape, ElemType, IsKMajor>;
                    constexpr auto swizzle_kind = ((Config == config_type::variable_atom_tile) and (ExecutionType == execution_type::device_pipeline)) ? swizzle_type::position_dependent : swizzle_type::position_independent;
                    return get_ldsm_swizzled_config<ldsm_instruction_t,
                                                    stsm_instruction_t,
                                                    AtomShape,
                                                    LayoutShape,
                                                    IsKMajor,
                                                    ElemType,
                                                    swizzle_kind>();
                }

                CUTE_GCC_UNREACHABLE;
            }

            template<bool             InstructionAvailable,
                     instruction_type InstructionType,
                     int              SM, // Fail if no instruction available
                     execution_type ExecutionType,
                     class TileShapeA,
                     bool IsAKMajor,
                     class TA,
                     int AlignmentA,
                     class TileShapeB,
                     bool IsBKMajor,
                     class TB,
                     int AlignmentB,
                     class TileShapeC,
                     bool IsCKMajor,
                     class TC,
                     int AlignmentC,
                     class WarpTile,
                     class SuperMMAInstruction>
            struct create_optimal_config {
                using a_layout = void;
                using b_layout = void;
                using c_layout = void;

                using a_load_op  = void;
                using b_load_op  = void;
                using c_load_op  = void;
                using c_store_op = void;

                using mma = void;

                static constexpr instruction_type instruction_kind = InstructionType;
                static constexpr bool             valid            = false;
            };

            template<int SM,
                     execution_type ExecutionType,
                     class TileShapeA,
                     bool IsAKMajor,
                     class TA,
                     int AlignmentA,
                     class TileShapeB,
                     bool IsBKMajor,
                     class TB,
                     int AlignmentB,
                     class TileShapeC,
                     bool IsCKMajor,
                     class TC,
                     int AlignmentC,
                     class WarpTile,
                     class SuperMMAInstruction>
            struct create_optimal_config<true,
                                         instruction_type::supermma,
                                         SM,
                                         ExecutionType,
                                         TileShapeA,
                                         IsAKMajor,
                                         TA,
                                         AlignmentA,
                                         TileShapeB,
                                         IsBKMajor,
                                         TB,
                                         AlignmentB,
                                         TileShapeC,
                                         IsCKMajor,
                                         TC,
                                         AlignmentC,
                                         WarpTile,
                                         SuperMMAInstruction> {
                // 1. Unpack inputs
                using instruction_shape                 = typename cute::MMA_Traits<SuperMMAInstruction>::Shape_MNK;
                static constexpr unsigned instruction_m = cute::tuple_element_t<0, instruction_shape>::value;
                static constexpr unsigned instruction_n = cute::tuple_element_t<1, instruction_shape>::value;
                static constexpr unsigned instruction_k = cute::tuple_element_t<2, instruction_shape>::value;

                static constexpr unsigned warp_m = cute::size<0>(WarpTile {});
                static constexpr unsigned warp_n = cute::size<1>(WarpTile {});


                // Intermediary A
                using atom_shape_a = cute::tuple<cute::Int<warp_m>, cute::Int<instruction_m>, cute::Int<instruction_k>>;
                using a_config =
                    decltype(get_swizzled_config<SM, atom_shape_a, TileShapeA, IsAKMajor, TA, AlignmentA, ExecutionType>());
                using a_layout     = cute::tuple_element_t<0, a_config>;
                using m_value_tile = cute::tuple_element_t<1, a_config>;
                using a_load_op    = cute::tuple_element_t<2, a_config>;

                // Intermediary B
                using atom_shape_b = cute::tuple<cute::Int<warp_n>, cute::Int<instruction_n>, cute::Int<instruction_k>>;
                // Reverse tile to reuse same function for A and B intermediary
                using reversed_tile_shape_b =
                    cute::Shape<cute::tuple_element_t<1, TileShapeB>, cute::tuple_element_t<0, TileShapeB>>;
                using b_config =
                    decltype(get_swizzled_config<SM, atom_shape_b, reversed_tile_shape_b, IsBKMajor, TB, AlignmentB, ExecutionType>());

                // Reverse back to K x N instead of N x K
                using b_layout     = decltype(transpose_first_two_modes_of_layout(cute::tuple_element_t<0, b_config> {}));
                using n_value_tile = cute::tuple_element_t<1, b_config>;
                using b_load_op    = cute::tuple_element_t<2, b_config>;

                // Intermediary C
                static constexpr int tiled_m_atom_size = cute::size(m_value_tile {}) / warp_m;
                static constexpr int tiled_n_atom_size = cute::size(n_value_tile {}) / warp_n;
                using atom_shape_c =
                    cute::tuple<cute::Int<1>, cute::Int<tiled_m_atom_size>, cute::Int<tiled_n_atom_size>>;
                using c_config         = decltype(get_swizzled_config<SM,
                                                                      atom_shape_c,
                                                                      TileShapeC,
                                                                      IsCKMajor,
                                                                      TC,
                                                                      AlignmentC,
                                                                      ExecutionType,
                                                                      config_type::fixed_atom_tile>());
                using c_layout         = cute::tuple_element_t<0, c_config>;
                using dummy_value_tile = cute::tuple_element_t<1, c_config>;
                using c_ldsm_op        = cute::tuple_element_t<2, c_config>;

                using c_stsm_op            = cute::tuple_element_t<3, c_config>;
                using autovectorizing_copy = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentC * 8>;
                static constexpr bool equal_size_precisions = (sizeof(TC) == sizeof(TA)) and (sizeof(TC) == sizeof(TB));
                using c_store_op = cute::conditional_t<equal_size_precisions, c_stsm_op, autovectorizing_copy>;
                using c_load_op  = cute::conditional_t<equal_size_precisions, c_ldsm_op, autovectorizing_copy>;

                // MMA --> Combine A and B value tile requirements
                using mma = decltype(cute::make_tiled_mma(
                    SuperMMAInstruction {},                                             // Instruction
                    WarpTile {},                                                        // ThreadTile
                    cute::Tile<m_value_tile, n_value_tile, cute::Int<instruction_k>> {} // ValueTile
                    ));

                static constexpr bool valid =
                    cute::is_composed_layout<a_layout>{} or cute::is_composed_layout<b_layout>{};
            };

            template<int SM,
                     class TileShapeA,
                     bool IsAKMajor,
                     class TA,
                     int AlignmentA,
                     class TileShapeB,
                     bool IsBKMajor,
                     class TB,
                     int AlignmentB,
                     class TileShapeC,
                     bool IsCKMajor,
                     class TC,
                     int AlignmentC,
                     class WarpGroupTile,
                     class GMMAInstruction>
            struct create_optimal_config<true,
                                         instruction_type::gmma,
                                         SM,
                                         execution_type::device_pipeline,
                                         TileShapeA,
                                         IsAKMajor,
                                         TA,
                                         AlignmentA,
                                         TileShapeB,
                                         IsBKMajor,
                                         TB,
                                         AlignmentB,
                                         TileShapeC,
                                         IsCKMajor,
                                         TC,
                                         AlignmentC,
                                         WarpGroupTile,
                                         GMMAInstruction> {
                static_assert(AlignmentA == 16 and AlignmentB == 16 and AlignmentC == 16);

                using a_layout = decltype(make_wgmma_compatible_layout<IsAKMajor, TA>(TileShapeA {}));
                using b_layout = decltype(make_wgmma_compatible_layout<not IsBKMajor, TB>(TileShapeB {}));
                using c_layout = decltype(cute::make_layout(TileShapeC{}, cute::conditional_return<IsCKMajor>(cute::LayoutRight{}, cute::LayoutLeft{})));

                // Not used in GMMA
                using a_load_op = void;
                using b_load_op = void;

                // TODO: integrate appropriate LDSM / STSM instructions for fp16 accumulators
                using c_load_op  = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentC * 8>;
                using c_store_op = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentC * 8>;

                using mma = decltype(cute::make_tiled_mma(GMMAInstruction {}, WarpGroupTile {}));

                static constexpr bool valid = true;
            };

            template<int SM,
                     class TileShapeA,
                     bool IsAKMajor,
                     class TA,
                     int AlignmentA,
                     class TileShapeB,
                     bool IsBKMajor,
                     class TB,
                     int AlignmentB,
                     class TileShapeC,
                     bool IsCKMajor,
                     class TC,
                     int AlignmentC,
                     class WarpGroupTile,
                     class UTCMMAInstruction>
            struct create_optimal_config<true,
                                         instruction_type::utcmma,
                                         SM,
                                         execution_type::device_pipeline,
                                         TileShapeA,
                                         IsAKMajor,
                                         TA,
                                         AlignmentA,
                                         TileShapeB,
                                         IsBKMajor,
                                         TB,
                                         AlignmentB,
                                         TileShapeC,
                                         IsCKMajor,
                                         TC,
                                         AlignmentC,
                                         WarpGroupTile,
                                         UTCMMAInstruction> {
                static_assert(AlignmentA == 16 and AlignmentB == 16 and AlignmentC == 16);

                using a_layout = decltype(make_wgmma_compatible_layout<IsAKMajor, TA>(TileShapeA {}));
                using b_layout = decltype(make_wgmma_compatible_layout<not IsBKMajor, TB>(TileShapeB {}));
                using c_layout = decltype(cute::make_layout(TileShapeC{}, cute::conditional_return<IsCKMajor>(cute::LayoutRight{}, cute::LayoutLeft{})));

                // Not used in UTCMMA
                using a_load_op = void;
                using b_load_op = void;

                // TODO: integrate appropriate LDSM / STSM instructions for fp16 accumulators
                using c_load_op  = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentC * 8>;
                using c_store_op = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentC * 8>;

                using mma = decltype(cute::make_tiled_mma(UTCMMAInstruction {}, WarpGroupTile {}));

                static constexpr bool valid = true;
            };

            // Struct for caching multiple constexpr accesses to
            // this information
            template<unsigned    Threads,
                     int         SM,
                     sm_modifier SMModifier,
                     class HasStreaming,
                     class HasWithPipeline,
                     class TA,
                     bool IsALayoutLeft,
                     int  AlignmentA,
                     class TB,
                     bool IsBLayoutLeft,
                     int  AlignmentB,
                     class TC,
                     bool IsCLayoutLeft,
                     int  AlignmentC,
                     int  M,
                     int  N,
                     int  K>
            struct optimal_config {
                // process input
                // =============
                static constexpr bool maximally_aligned_ab = AlignmentA == 16 and AlignmentB == 16;

                using tile_shape_a                 = cute::Shape<cute::Int<M>, cute::Int<K>>;
                static constexpr bool is_a_k_major = not IsALayoutLeft;
                using tile_shape_b                 = cute::Shape<cute::Int<K>, cute::Int<N>>;
                static constexpr bool is_b_k_major = IsBLayoutLeft;
                using tile_shape_c                 = cute::Shape<cute::Int<M>, cute::Int<N>>;
                static constexpr bool is_c_k_major = not IsCLayoutLeft;

                // currently only superMMA is supported, where a single warp is the execution unit
                static constexpr bool valid_block_dim = is_power_of_2(Threads) and Threads >= 32 and Threads <= 1024;
                static constexpr int  num_warps       = cute::ceil_div(Threads, 32);
                using gemm_shape                      = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;

                using supermma = get_best_supermma_tiled_instruction<SM, num_warps, gemm_shape, TA, TB, TC>;
                static constexpr bool is_supermma_available = valid_block_dim and
                                                              not cute::is_void_v<typename supermma::instruction> and
                                                              not cute::is_void_v<typename supermma::tile>;

                using gmma =
                    get_best_gmma_tiled_instruction<SM, num_warps, gemm_shape, TA, TB, TC, is_a_k_major, is_b_k_major>;

                static constexpr bool is_gmma_available = 
                    HasWithPipeline{} and HasStreaming {} and (SM == 900) and SMModifier == sm_modifier::arch_specific and valid_block_dim and
                    maximally_aligned_ab and not cute::is_void_v<typename gmma::instruction> and
                    not cute::is_void_v<typename gmma::tile>;

                using utcmma = get_best_utcmma_tiled_instruction<SM,
                                                                 num_warps,
                                                                 gemm_shape,
                                                                 TA,
                                                                 TB,
                                                                 TC,
                                                                 is_a_k_major,
                                                                 is_b_k_major>;
                static constexpr bool is_utcmma_available =
                    HasWithPipeline {} and HasStreaming {} and
                    (SM == 1000 or SM == 1030 or SM == 1100) and
                    // No S8 UTCMMA on Blackwell Ultra
                    (SM != 1030 or !commondx::is_integral_v<TC>) and
                    SMModifier == sm_modifier::arch_specific and (Threads == 128) and maximally_aligned_ab and
                    not cute::is_void_v<typename utcmma::instruction> and not cute::is_void_v<typename utcmma::tile>;

                // Decide on which instruction to use
                using nonportable                              = cute::conditional_t<is_utcmma_available, utcmma, gmma>;
                static constexpr bool is_nonportable_available = is_utcmma_available or is_gmma_available;
                using result_choice = cute::conditional_t<is_nonportable_available, nonportable, supermma>;

                using instruction                                  = typename result_choice::instruction;
                using tile                                         = typename result_choice::tile;
                static constexpr instruction_type instruction_kind = result_choice::instruction_kind;
                static constexpr bool             available        = is_nonportable_available or is_supermma_available;
                static_assert(not(cute::is_void_v<instruction> xor cute::is_void_v<tile>));

                // intermediary
                // ============
                static constexpr execution_type exec = HasWithPipeline{} ? execution_type::device_pipeline : execution_type::cta_tile;
                using config = create_optimal_config<available,
                                                     instruction_kind,
                                                     SM,
                                                     exec,
                                                     tile_shape_a,
                                                     is_a_k_major,
                                                     TA,
                                                     AlignmentA,
                                                     tile_shape_b,
                                                     is_b_k_major,
                                                     TB,
                                                     AlignmentB,
                                                     tile_shape_c,
                                                     is_c_k_major,
                                                     TC,
                                                     AlignmentC,
                                                     tile,
                                                     instruction>;

                // return output
                // =============

                using a_layout  = typename config::a_layout;
                using b_layout  = typename config::b_layout;
                using c_layout  = typename config::c_layout;
                using tiled_mma = typename config::mma;

                using a_copy_op             = typename config::a_load_op;
                using b_copy_op             = typename config::b_load_op;
                using c_copy_load_op        = typename config::c_load_op;
                using c_copy_store_op       = typename config::c_store_op;
                static constexpr bool valid = config::valid;
            };

            template<int         Matrix, // 0: A, 1: B, 2: C
                     int         BlockDim,
                     int         SM,
                     sm_modifier SMModifier,
                     class HasStreaming,
                     class HasWithPipeline,
                     typename TA,
                     bool IsALayoutLeft,
                     int  AlignmentA,
                     typename TB,
                     bool IsBLayoutLeft,
                     int  AlignmentB,
                     typename TC,
                     bool IsCLayoutLeft,
                     int  AlignmentC,
                     int  M,
                     int  N,
                     int  K>
            struct optimal_config_entry {
                using config = optimal_config<BlockDim,
                                              SM,
                                              SMModifier,
                                              HasStreaming,
                                              HasWithPipeline,
                                              TA,
                                              IsALayoutLeft,
                                              AlignmentA,
                                              TB,
                                              IsBLayoutLeft,
                                              AlignmentB,
                                              TC,
                                              IsCLayoutLeft,
                                              AlignmentC,
                                              M,
                                              N,
                                              K>;
                CUBLASDX_HOST_DEVICE
                static constexpr auto get() {
                    auto ret = cute::get<Matrix>(cute::make_tuple(
                        typename config::a_layout {}, typename config::b_layout {}, typename config::c_layout {}));
                    return ret;
                }

                static constexpr bool is_valid() { return config::valid; }
            };
        } // namespace layout_database
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_SUGGESTED_LAYOUTS_HPP
