// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_DATABASE_GETTER_HPP
#define CUBLASDX_DATABASE_DATABASE_GETTER_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "commondx/device_info.hpp"
#include "commondx/type_list.hpp"

#include "cublasdx/detail/numerical.hpp"
#include "cublasdx/detail/cute/cute_tensor.hpp"

#include "cublasdx/database/gemm/mma_atoms.hpp"
#include "cublasdx/database/gemm/database_records.hpp"
#include "cublasdx/database/gemm/database_workarounds.hpp"
#include "cublasdx/database/gemm/database_redirect.hpp"
#include "cublasdx/database/gemm/suggested/suggested_instructions.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {

            using cute::_1;
            using cute::Int;
            using cute::Layout;
            using cute::Shape;
            using cute::Stride;

            using cublasdx::detail::layout_database::get_best_ldsm_instruction_t;
            using cublasdx::detail::layout_database::get_best_stsm_instruction_t;

            // Selects generated_config from commondx::type_list based on blockdim,
            // if there is no such implementation in list search_by_blockdim::type is set to void.
            template<int ThreadsAvailable, typename ImplementationList>
            struct search_by_blockdim;

            template<int ThreadsAvailable, typename GeneratedConfig>
            struct search_by_blockdim<ThreadsAvailable, commondx::type_list<GeneratedConfig>> {
                using type = cute::conditional_t<GeneratedConfig::blockdim == ThreadsAvailable, GeneratedConfig, void>;
            };

            template<int ThreadsAvailable, typename GeneratedConfig, typename... RemainingConfigs>
            struct search_by_blockdim<ThreadsAvailable, commondx::type_list<GeneratedConfig, RemainingConfigs...>> {
                using type = cute::conditional_t<
                    GeneratedConfig::blockdim == ThreadsAvailable,
                    GeneratedConfig,
                    typename search_by_blockdim<ThreadsAvailable, commondx::type_list<RemainingConfigs...>>::type>;
            };

            template<bool HasBlockDim, int ThreadsAvailable, typename ConfigList>
            using search_struct = cute::conditional_t<HasBlockDim,
                                                      search_by_blockdim<ThreadsAvailable, ConfigList>,
                                                      commondx::type_list_element<0, ConfigList>>;

            CUBLASDX_HOST_DEVICE
            constexpr auto fallback_tile_config(int num_threads) {
                switch (num_threads) {
                    case 1024: {
                        return cute::make_tuple(mma_atom::universal_fma, 32, 32);
                    }
                    case 512: {
                        return cute::make_tuple(mma_atom::universal_fma, 32, 16);
                    }
                    case 256: {
                        return cute::make_tuple(mma_atom::universal_fma, 16, 16);
                    }
                    case 128: {
                        return cute::make_tuple(mma_atom::universal_fma, 16, 8);
                    }
                    case 64: {
                        return cute::make_tuple(mma_atom::universal_fma, 8, 8);
                    }
                    case 32: {
                        return cute::make_tuple(mma_atom::universal_fma, 8, 4);
                    }
                    default: {
                        return cute::make_tuple(mma_atom::universal_fma, num_threads, 1);
                    }
                }

                CUTE_GCC_UNREACHABLE;
            }

            // RETURN TYE STD::TUPLE CONTAINS 3 ELEMENTS:
            // 1. MMA_ATOM, Enum value pointing to which atom should be used
            // 2. INT, TileX size
            // 3. INT, TileY size
            // Types passed as AType, BType, CType are CUTLASS types
            // e.g. cute::half_t and not __half
            template<typename AType,
                     typename BType,
                     typename CType,
                     int M,
                     int N,
                     int K,
                     class SMOperator,
                     bool     AreThreadsSpecified,
                     unsigned Threads>
            constexpr auto get_tile_config() {
#ifdef __NVCOMPILER
#    pragma diag_suppress 185
#    pragma diag_suppress 111
#endif
                using SM = cublasdx::SM<SMOperator::value>;
                // Decay datatypes to ones supported by the database, many -> one
                using atype_db = database_precision_redirect_a_t<AType, BType, CType, SM>;
                using btype_db = database_precision_redirect_b_t<AType, BType, CType, SM>;
                using ctype_db = database_precision_redirect_c_t<AType, BType, CType, SM>;

                constexpr int averaged_size = lower_bound_int_sqrt(M * N);

                if constexpr (M == 1 or N == 1 or K == 1) {
                    constexpr int fallback_threads =
                        AreThreadsSpecified ? Threads
                                            : cute::max(32, cute::min(1024, closest_next_power_of_two(averaged_size)));
                    return fallback_tile_config(fallback_threads);
                }

                // Lambda which checks if record from entry list matches and is usable
                // in necessary circumstances
                const auto database_lookup = [](auto query) constexpr {
                    if constexpr (decltype(query)::defined) {
                        // 1. Pick entry from DB for "downcased" precision, to be later brought back to original types
                        using entry =
                            typename search_struct<AreThreadsSpecified, Threads, typename decltype(query)::list>::type;
                        if constexpr (!cute::is_same_v<entry, void>) {
                            constexpr auto converted_mma = map_mma_from_database<AType, BType, CType, SM>(entry::mma);
                            if (converted_mma ==
                                apply_mma_workarounds(converted_mma, M, N, K, entry::blockdim, SM::value)) {
                                constexpr auto inner_tiles = entry::tiles;
                                return cute::make_tuple(true,
                                                        cute::make_tuple(
                                                            // 2. Map MMA atom from DB entry to original types
                                                            converted_mma,
                                                            cute::get<0>(inner_tiles),
                                                            cute::get<1>(inner_tiles)));
                            }
                        }
                    }

                    return cute::make_tuple(false, cute::make_tuple(mma_atom::universal_fma, 0, 0));
                };

                // To take both M and N under account use the square root of their product
                // as database entry size (database only has square records inside itself)
                using mmm_query = database::
                    generated_config<atype_db, btype_db, ctype_db, averaged_size, averaged_size, averaged_size, SM>;

                // No rounding
                auto mmm_entry = database_lookup(mmm_query {});
                if (cute::get<0>(mmm_entry)) {
                    return cute::get<1>(mmm_entry);
                }

                // Rounding to 4 / 8 / (16 * n)
                constexpr int rounded = (averaged_size < 4 and K < 128) ? 4 : (averaged_size < 8 and K < 128) ? 8 : closest_multiple_of<16>(averaged_size);
                using rounded_mmm_query =
                    database::generated_config<atype_db, btype_db, ctype_db, rounded, rounded, rounded, SM>;
                auto rounded_mmm_entry = database_lookup(rounded_mmm_query {});
                if (cute::get<0>(rounded_mmm_entry)) {
                    return cute::get<1>(rounded_mmm_entry);
                }

                // If no record present in the database try finding a matching config with the heuristic
                constexpr auto default_mma = get_default_mma<AType, BType, CType, SM::value>();
                constexpr int  heuristic_threads =
                    AreThreadsSpecified ? Threads : cute::min(1024, cute::max(32, closest_next_power_of_two(averaged_size)));

                constexpr auto selected_mma = apply_mma_workarounds(default_mma, M, N, K, heuristic_threads, SM::value);
                using selected_mma_t        = decltype(convert_mma_atom_to_cute<AType, BType, CType, selected_mma>());
                constexpr int thread_atom =
                    decltype(cute::size(typename cute::MMA_Traits<selected_mma_t>::ThrID {}))::value;

                const auto size_tile_x =
                    decltype(cute::size<0>(typename cute::MMA_Traits<selected_mma_t>::Shape_MNK {}))::value;
                const auto size_tile_y =
                    decltype(cute::size<1>(typename cute::MMA_Traits<selected_mma_t>::Shape_MNK {}))::value;

                constexpr int units_to_divide = heuristic_threads / thread_atom;

                for (int tile_x = 1; tile_x <= units_to_divide; ++tile_x) {
                    int tile_y = units_to_divide / tile_x;
                    // This loop looks for such a division of ThreadsAvailable into TileX
                    // and TileY that ThreadsAvailable == TileX * TileY and TileX > TileY
                    // keeping TileX as small as possible. Ideally it will find square root
                    // of ThreadsAvailable, but otherwise the most square combination
                    // possible
                    if (tile_x >= tile_y && tile_x * tile_y == units_to_divide &&
                        (tile_x * tile_y * thread_atom) >= 32 &&
                        ((M > (size_tile_x * tile_x)) ||
                         ((size_tile_x * tile_x > M) && (size_tile_x * tile_x - M < size_tile_x))) &&
                        ((N > (size_tile_y * tile_y)) ||
                         ((size_tile_y * tile_y > N) && (size_tile_y * tile_y - N < size_tile_y)))) {
                        return cute::make_tuple(selected_mma, tile_x, tile_y);
                    }
                }
#ifdef __NVCOMPILER
#    pragma diag_warning 185
#    pragma diag_warning 111
#endif

                // Final fallback
                return fallback_tile_config(heuristic_threads);
            }

            // Types passed here are CUTLASS types
            //
            template<typename AType,
                     typename BType,
                     typename CType,
                     int M,
                     int N,
                     int K,
                     class SMOperator,
                     sm_modifier SMModifier,
                     class BlockSize,
                     class OverloadedTileOperator = cublasdx::experimental::Tile<void, 0, 0, void>>
            struct get_database_threads {
            private:
                using SM = cublasdx::SM<SMOperator::value>;

                static constexpr bool is_tile_overloaded = OverloadedTileOperator::valid;

                static constexpr bool are_threads_specified = not cute::is_same_v<BlockSize, void>;
                using specified_threads_t = cute::conditional_t<are_threads_specified, BlockSize, BlockDim<0>>;

                static constexpr auto tile_config = get_tile_config<AType,
                                                                    BType,
                                                                    CType,
                                                                    M,
                                                                    N,
                                                                    K,
                                                                    SM,
                                                                    are_threads_specified,
                                                                    specified_threads_t::flat_size>();

                static constexpr auto db_atom_v = cute::get<0>(tile_config);
                static constexpr auto db_tile_m = cute::get<1>(tile_config);
                static constexpr auto db_tile_n = cute::get<2>(tile_config);

                using cute_atom_t =
                    cute::conditional_t<is_tile_overloaded,
                                        typename OverloadedTileOperator::mma,
                                        decltype(convert_mma_atom_to_cute<AType, BType, CType, db_atom_v>())>;
                static constexpr auto cute_atom = cute_atom_t {};

                static constexpr int tile_m =
                    cute::conditional_return<is_tile_overloaded>(OverloadedTileOperator::tile_x, db_tile_m);
                static constexpr int tile_n =
                    cute::conditional_return<is_tile_overloaded>(OverloadedTileOperator::tile_y, db_tile_n);
                static constexpr auto instruction_threads =
                    typename cute::MMA_Traits<cute::decay_t<decltype(cute_atom)>>::ThrID {};

            public:
                using type = Shape<Int<(round_up(cute::size(instruction_threads) * tile_m * tile_n, 32))>>;
                static constexpr int value = cute::size(type {});
            };

            template<typename AType,
                     typename BType,
                     typename CType,
                     int M,
                     int N,
                     int K,
                     class SM,
                     sm_modifier SMModifier,
                     class BlockSize,
                     class OverloadedTileOperator = cublasdx::experimental::Tile<void, 0, 0, void>>
            using get_database_threads_t = typename get_database_threads<AType,
                                                                         BType,
                                                                         CType,
                                                                         M,
                                                                         N,
                                                                         K,
                                                                         SM,
                                                                         SMModifier,
                                                                         BlockSize,
                                                                         OverloadedTileOperator>::type;

            template<class LDSMInstruction, class AtomShape, class ElemType>
            struct ldsm_tile_helper {
                static constexpr int warp_m = cute::get<0>(AtomShape {});
                static constexpr int atom_m = cute::get<1>(AtomShape {});
                static constexpr int atom_k = cute::get<2>(AtomShape {});

                static constexpr int thread_adjusted_atom_length = warp_m * atom_m;
                static constexpr int ldsm_bytes =
                    cute::size(typename cute::Copy_Traits<LDSMInstruction>::RefLayout {}) / 8;
                using type =
                    Int<cute::ceil_div(ldsm_bytes, atom_m* atom_k * sizeof(ElemType)) * thread_adjusted_atom_length>;
            };

            template<class AtomShape, class ElemType>
            struct ldsm_tile_helper<void, AtomShape, ElemType> {
                using type = void;
            };

            enum class lds_tile_type
            {
                regular_mma,
                vectorized_mma
            };

            template<int PermutationLevel, class AtomShape, lds_tile_type TileType, class = void>
            struct lds_tile_helper {
                static_assert(PermutationLevel > 0);
                static constexpr int warp_m = cute::get<0>(AtomShape {});
                static constexpr int atom_m = cute::get<1>(AtomShape {});

                using type =
                    Layout<Shape<Int<warp_m * atom_m>, Int<PermutationLevel>>, Stride<Int<PermutationLevel>, _1>>;
            };

            template<int PermutationLevel, class AtomShape>
            struct lds_tile_helper<PermutationLevel,
                                   AtomShape,
                                   lds_tile_type::vectorized_mma,
                                   cute::enable_if_t<(PermutationLevel > 1)>> {
                static constexpr int warp_m = cute::get<0>(AtomShape {});
                static constexpr int atom_m = cute::get<1>(AtomShape {});

                using type = Layout<Shape<Int<atom_m>, Int<warp_m>, Int<PermutationLevel>>,
                                    Stride<_1, Int<PermutationLevel * atom_m>, Int<atom_m>>>;
            };

            template<class AtomShape, lds_tile_type TileType>
            struct lds_tile_helper<1, AtomShape, TileType> {
                static constexpr int warp_m = cute::get<0>(AtomShape {});
                static constexpr int atom_m = cute::get<1>(AtomShape {});

                using type = Layout<Shape<Int<warp_m * atom_m>>, Stride<_1>>;
            };

            template<typename AType,
                     typename BType,
                     typename CType,
                     int         M,
                     int         N,
                     int         K,
                     arrangement AArr,
                     arrangement BArr,
                     arrangement CArr,
                     int         AAlignment,
                     int         BAlignment,
                     int         CAlignment,
                     class SM,
                     class BlockSize,
                     class OverloadedTileOperator = cublasdx::experimental::Tile<void, 0, 0, void>>
            struct get_database_config {
            private:
                static constexpr bool is_tile_overloaded = OverloadedTileOperator::valid;

                static constexpr bool is_a_col_major = (AArr == arrangement::col_major);
                static constexpr bool is_b_row_major = (BArr == arrangement::row_major);
                static constexpr bool is_c_col_major = (CArr == arrangement::col_major);

                static constexpr bool are_threads_specified = not cute::is_same_v<BlockSize, void>;
                using specified_threads_t = cute::conditional_t<are_threads_specified, BlockSize, BlockDim<0>>;

                static constexpr auto tile_config = get_tile_config<AType,
                                                                    BType,
                                                                    CType,
                                                                    M,
                                                                    N,
                                                                    K,
                                                                    SM,
                                                                    are_threads_specified,
                                                                    specified_threads_t::flat_size>();

                static constexpr bool make_config_row_major = (not is_a_col_major and is_b_row_major);

                static constexpr auto db_retrieved_atom_v = cute::get<0>(tile_config);
                // If necessary, based on transpose, switch instruction flow from col major to row major
                static constexpr bool is_vectorized_fma_v =
                    is_tile_overloaded ? is_vectorized_fma<typename OverloadedTileOperator::mma>()
                                       : is_vectorized_fma(db_retrieved_atom_v);
                static constexpr auto db_atom_v = []() {
                    if constexpr (is_vectorized_fma_v and make_config_row_major) {
                        return convert_vector_fma_to_row_major<db_retrieved_atom_v>::value;
                    } else {
                        return db_retrieved_atom_v;
                    }
                    CUTE_GCC_UNREACHABLE;
                }();

                static constexpr auto db_tile_m = cute::get<1>(tile_config);
                static constexpr auto db_tile_n = cute::get<2>(tile_config);

                using cute_atom_t =
                    cute::conditional_t<is_tile_overloaded,
                                        typename OverloadedTileOperator::mma,
                                        decltype(convert_mma_atom_to_cute<AType, BType, CType, db_atom_v>())>;
                static constexpr auto cute_atom = cute_atom_t {};
                static constexpr auto mma_tile_m =
                    cute::conditional_return<is_tile_overloaded>(OverloadedTileOperator::tile_x,
                                                                 make_config_row_major ? db_tile_n : db_tile_m);
                static constexpr auto mma_tile_n =
                    cute::conditional_return<is_tile_overloaded>(OverloadedTileOperator::tile_y,
                                                                 make_config_row_major ? db_tile_m : db_tile_n);
                static constexpr bool is_supermma = cute::size(typename cute::MMA_Traits<cute_atom_t>::ThrID {}) == 32;

                static constexpr auto instruction_shape =
                    typename cute::MMA_Traits<cute::decay_t<decltype(cute_atom)>>::Shape_MNK {};
                static constexpr int atom_m = cute::get<0>(instruction_shape);
                static constexpr int atom_n = cute::get<1>(instruction_shape);
                static constexpr int atom_k = cute::get<2>(instruction_shape);

                static constexpr auto perm_m = mma_tile_m * atom_m;
                static constexpr auto perm_n = mma_tile_n * atom_n;
                static constexpr auto perm_k = atom_k;

                // ============================
                // Find copy instruction for A
                // ============================

                // Attempt LDSM for A
                using a_layout_shape = Shape<Int<M>, Int<K>>;
                using a_atom_shape   = Shape<Int<mma_tile_m>, Int<atom_m>, Int<atom_k>>;
                using a_ldsm_instruction_t =
                    get_best_ldsm_instruction_t<SM::value, a_layout_shape, a_atom_shape, AType, not is_a_col_major>;
                using a_ldsm_tile_t = typename ldsm_tile_helper<a_ldsm_instruction_t, a_atom_shape, AType>::type;

                // Attempt LDS vectorization for A
                static constexpr bool is_a_instruction_col_vectorized =
                    cute::is_same_v<cute_atom_t, cublasdx::detail::SM100_2x1x1_F32F32F32F32>;
                static constexpr int tiled_atom_shape_m_first  = mma_tile_m * cute::get<0>(instruction_shape);
                static constexpr int tiled_atom_shape_m_second = cute::get<2>(instruction_shape);
                using tiled_atom_shape_m = Shape<Int<tiled_atom_shape_m_first>, Int<tiled_atom_shape_m_second>>;

                static constexpr bool compat_m =
                    is_a_col_major and cute::evenly_divides(cute::Shape<Int<M>, Int<K>> {}, tiled_atom_shape_m {});
                static constexpr auto max_repeated_atoms_m =
                    cute::ceil_div(M, (mma_tile_m * cute::get<0>(instruction_shape)));
                static constexpr auto a_lds_load_size =
                    is_vectorized_fma_v ? sizeof(typename decltype(cute_atom)::ARegisters) : sizeof(AType);
                static constexpr auto mma_vec_m =
                    compat_m ? cute::gcd(max_repeated_atoms_m, cute::ceil_div(AAlignment, a_lds_load_size)) : 1;
                using a_lds_instruction_t = cute::AutoVectorizingCopyWithAssumedAlignment<AAlignment * 8>;
                using a_lds_tile_t =
                    typename lds_tile_helper<mma_vec_m,
                                             a_atom_shape,
                                             is_a_instruction_col_vectorized ? lds_tile_type::vectorized_mma
                                                                             : lds_tile_type::regular_mma>::type;

                // Finalize A copy instruction and M value tile modification
                static constexpr bool a_ldsm_condition =
                    (not cute::is_same_v<a_ldsm_instruction_t, void>) and (AAlignment == 16) and is_supermma;
                using a_instruction_t =
                    cute::conditional_t<a_ldsm_condition, a_ldsm_instruction_t, a_lds_instruction_t>;
                using m_value_tile = cute::conditional_t<a_ldsm_condition, a_ldsm_tile_t, a_lds_tile_t>;

                // ============================
                // Find copy instruction for B
                // ============================

                // Attempt LDSM for B
                using b_layout_shape = Shape<Int<N>, Int<K>>;
                using b_atom_shape   = Shape<Int<mma_tile_n>, Int<atom_n>, Int<atom_k>>;
                using b_ldsm_instruction_t =
                    get_best_ldsm_instruction_t<SM::value, b_layout_shape, b_atom_shape, BType, not is_b_row_major>;

                using b_ldsm_tile_t = typename ldsm_tile_helper<b_ldsm_instruction_t, b_atom_shape, BType>::type;

                // Attempt LDS vectorization for B
                static constexpr bool is_b_instruction_row_vectorized =
                    cute::is_same_v<cute_atom_t, cublasdx::detail::SM100_1x2x1_F32F32F32F32>;
                static constexpr int tiled_atom_shape_n_first  = mma_tile_n * cute::get<1>(instruction_shape);
                static constexpr int tiled_atom_shape_n_second = cute::get<2>(instruction_shape);
                using tiled_atom_shape_n = Shape<Int<tiled_atom_shape_n_first>, Int<tiled_atom_shape_n_second>>;
                static constexpr bool compat_n =
                    is_b_row_major and cute::evenly_divides(cute::Shape<Int<N>, Int<K>> {}, tiled_atom_shape_n {});
                static constexpr auto max_repeated_atoms_n =
                    cute::ceil_div(N, (mma_tile_n * cute::get<1>(instruction_shape)));
                static constexpr auto b_lds_load_size =
                    is_vectorized_fma_v ? sizeof(typename decltype(cute_atom)::BRegisters) : sizeof(BType);
                static constexpr auto mma_vec_n =
                    compat_n ? cute::gcd(max_repeated_atoms_n, cute::ceil_div(BAlignment, b_lds_load_size)) : 1;
                using b_lds_instruction_t = cute::AutoVectorizingCopyWithAssumedAlignment<BAlignment * 8>;
                using b_lds_tile_t =
                    typename lds_tile_helper<mma_vec_n,
                                             b_atom_shape,
                                             is_b_instruction_row_vectorized ? lds_tile_type::vectorized_mma
                                                                             : lds_tile_type::regular_mma>::type;

                // Finalize B copy instruction and N value tile modification
                static constexpr bool b_ldsm_condition =
                    (not cute::is_same_v<b_ldsm_instruction_t, void>) and (BAlignment == 16) and is_supermma;
                using b_instruction_t =
                    cute::conditional_t<b_ldsm_condition, b_ldsm_instruction_t, b_lds_instruction_t>;
                using n_value_tile = cute::conditional_t<b_ldsm_condition, b_ldsm_tile_t, b_lds_tile_t>;

                // ============================
                // Find copy instructions for C
                // ============================

                // Attempt LDSM for C
                static_assert(cute::size(m_value_tile {}) % mma_tile_m == 0);
                static_assert(cute::size(n_value_tile {}) % mma_tile_n == 0);
                using c_tiled_atom_shape   = Shape<Int<1>,
                                                   Int<cute::size(m_value_tile {}) / mma_tile_m>,
                                                   Int<cute::size(n_value_tile {}) / mma_tile_n>>;
                using limiting_shape       = decltype(cute::select<1, 2>(c_tiled_atom_shape {}));
                using c_ldsm_instruction_t = get_best_ldsm_instruction_t<SM::value,
                                                                         limiting_shape,
                                                                         c_tiled_atom_shape,
                                                                         CType,
                                                                         not is_c_col_major>;
                using c_stsm_instruction_t = get_best_stsm_instruction_t<SM::value,
                                                                         limiting_shape,
                                                                         c_tiled_atom_shape,
                                                                         CType,
                                                                         not is_c_col_major>;

                // Attempt LDS vectorization for B
                using c_lds_instruction_t = cute::AutoVectorizingCopyWithAssumedAlignment<CAlignment * 8>;

                // Finalize B copy instruction and N value tile modification
                static constexpr bool equal_size_precisions =
                    (sizeof(AType) == sizeof(BType)) and (sizeof(AType) == sizeof(CType));

                static constexpr bool c_ldsm_condition = equal_size_precisions and
                                                         (not cute::is_same_v<c_ldsm_instruction_t, void>) and
                                                         (CAlignment == 16) and is_supermma;
                using c_load_instruction_t =
                    cute::conditional_t<c_ldsm_condition, c_ldsm_instruction_t, c_lds_instruction_t>;

                static constexpr bool c_stsm_condition = equal_size_precisions and
                                                         (not cute::is_same_v<c_stsm_instruction_t, void>) and
                                                         (CAlignment == 16) and is_supermma;
                using c_store_instruction_t =
                    cute::conditional_t<c_stsm_condition, c_stsm_instruction_t, c_lds_instruction_t>;

                // ===========================
                // Finalize tiled_mma creation
                // ===========================

                static constexpr auto thread_layout_shape = Shape<Int<mma_tile_m>, Int<mma_tile_n>> {};
                static constexpr auto thread_layout_stride_atom =
                    cute::conditional_t<make_config_row_major, cute::LayoutRight, cute::LayoutLeft> {};

                static constexpr auto mma_instruction = cute_atom;
                static constexpr auto mma_thread_tile =
                    cute::make_layout(thread_layout_shape, thread_layout_stride_atom);
                static constexpr auto mma_value_tile =
                    cute::Tile<m_value_tile, n_value_tile, Layout<Shape<Int<perm_k>>, Stride<_1>>> {};

                // RETURN VALUES
            public:
                using type           = decltype(cute::make_tiled_mma(mma_instruction, mma_thread_tile, mma_value_tile));
                using a_copy_op      = a_instruction_t;
                using b_copy_op      = b_instruction_t;
                using c_copy_load_op = c_load_instruction_t;
                using c_copy_store_op = c_store_instruction_t;
            };

            template<typename AType,
                     typename BType,
                     typename CType,
                     int         M,
                     int         N,
                     int         K,
                     arrangement AArr,
                     arrangement BArr,
                     arrangement CArr,
                     int         AAlignment,
                     int         BAlignment,
                     int         CAlignment,
                     class SM,
                     class BlockSize>
            using get_database_config_t = typename get_database_config<AType,
                                                                       BType,
                                                                       CType,
                                                                       M,
                                                                       N,
                                                                       K,
                                                                       AArr,
                                                                       BArr,
                                                                       CArr,
                                                                       AAlignment,
                                                                       BAlignment,
                                                                       CAlignment,
                                                                       SM,
                                                                       BlockSize>::type;

        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_DATABASE_GETTER_HPP
