// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_SUGGESTED_UTCMMA_HPP
#define CUBLASDX_DATABASE_SUGGESTED_UTCMMA_HPP

#include "cublasdx/database/gemm/suggested/common.hpp"

namespace cublasdx {
    namespace detail {
        namespace layout_database {
            CUBLASDX_HOST_DEVICE
            constexpr int get_utcmma_n(int tile_n) {
                for (int current_n = 256; current_n >= 8; current_n -= 8) {
                    if (tile_n % current_n == 0)
                        return current_n;
                }

                return 1;
            }

            CUBLASDX_HOST_DEVICE
            constexpr int get_fp8_utcmma_n(int tile_n) {
                for (int current_n = 256; current_n >= 16; current_n -= 16) {
                    if (tile_n % current_n == 0)
                        return current_n;
                }

                return 1;
            }

            template<int M, int N, int K, class TA, class TB, class TC, bool IsAKMajor, bool IsBKMajor>
            struct get_best_utcmma_instruction {
                using instruction = void;
            };

            // TMEM holds 128 lanes x 512 columns of 4 bytes per SM. A 1-CTA UTCMMA
            // accumulator uses one column range per 128-row M block, so an M x N
            // accumulator needs (M / instruction_m) x pow2ceil(N * sizeof(C) / 4)
            // columns (32-column minimum granularity, matching the TMEM
            // accumulator's stage sizing). Reject instructions whose accumulator
            // cannot fit at least one full stage.
            CUBLASDX_HOST_DEVICE
            constexpr bool utcmma_tmem_fits(int m, int instruction_m, int n, int sizeof_c) {
                constexpr int tmem_columns = 512;
                const int m_blocks = m / instruction_m;
                const int cols     = n * sizeof_c / 4;
                int stage_cols = 32;
                while (stage_cols < cols) {
                    stage_cols *= 2;
                }
                return m_blocks * stage_cols <= tmem_columns;
            }

            CUBLASDX_HOST_DEVICE
            constexpr bool is_utcmma_n8_compatible(int n) {
                return (n >= 8 and n <= 256) and (n % 8 == 0);
            }

            CUBLASDX_HOST_DEVICE
            constexpr bool is_utcmma_n16_compatible(int n) {
                return (n >= 16 and n <= 256) and (n % 16 == 0);
            }

            CUBLASDX_HOST_DEVICE
            constexpr bool is_utcmma_s8_n_compatible(int n) {
                return (n == 8) or is_utcmma_n16_compatible(n);
            }

            CUBLASDX_HOST_DEVICE
            constexpr bool is_utcmma_fp8_n_compatible(int n, bool is_b_k_major) {
                return is_b_k_major ? is_utcmma_n8_compatible(n) : is_utcmma_n16_compatible(n);
            }

#define UTCMMA_DEFINITION(APrecision, BPrecision, CPrecision, InstructionName)                              \
    template<int M, int N, int K, bool IsAKMajor, bool IsBKMajor>                                           \
    struct get_best_utcmma_instruction<M, N, K, APrecision, BPrecision, CPrecision, IsAKMajor, IsBKMajor> { \
        static constexpr int  dummy_m       = 64;                                                           \
        static constexpr int  dummy_n       = 64;                                                           \
        static constexpr bool m_compatible  = (M % 128 == 0) or (M == 64);                                  \
        static constexpr int  instruction_m = m_compatible ? ((M % 128 == 0) ? 128 : 64) : dummy_m;         \
        static constexpr bool n_compatible  = is_utcmma_n8_compatible(N);                                  \
        static constexpr int  instruction_n = n_compatible ? N : dummy_n;                                   \
        static constexpr int  instruction_k = 256 / cute::sizeof_bits<APrecision>::value;                   \
        static constexpr bool k_compatible  = K % instruction_k == 0;                                       \
        static constexpr bool tmem_compatible =                                                             \
            utcmma_tmem_fits(M, instruction_m, N, static_cast<int>(sizeof(CPrecision)));                    \
        static constexpr bool valid = m_compatible and n_compatible and k_compatible and tmem_compatible;   \
        using instruction =                                                                                 \
            cute::conditional_t<valid,                                                                      \
                                InstructionName<APrecision,                                                 \
                                                BPrecision,                                                 \
                                                CPrecision,                                                 \
                                                instruction_m,                                              \
                                                instruction_n,                                              \
                                                IsAKMajor ? cute::UMMA::Major::K : cute::UMMA::Major::MN,   \
                                                IsBKMajor ? cute::UMMA::Major::K : cute::UMMA::Major::MN>,  \
                                void>;                                                                      \
    };

#define UTCMMA_TF32_DEFINITION(APrecision, BPrecision, CPrecision, InstructionName)                      \
    template<int M, int N, int K>                                                                        \
    struct get_best_utcmma_instruction<M, N, K, APrecision, BPrecision, CPrecision, true, true> {        \
        static constexpr int  dummy_m       = 64;                                                        \
        static constexpr int  dummy_n       = 64;                                                        \
        static constexpr bool m_compatible  = (M % 128 == 0) or (M == 64);                               \
        static constexpr int  instruction_m = m_compatible ? ((M % 128 == 0) ? 128 : 64) : dummy_m;      \
        static constexpr bool n_compatible  = is_utcmma_n8_compatible(N);                               \
        static constexpr int  instruction_n = n_compatible ? N : dummy_n;                                \
        static constexpr int  instruction_k = 256 / cute::sizeof_bits<APrecision>::value;                \
        static constexpr bool k_compatible  = K % instruction_k == 0;                                    \
        static constexpr bool tmem_compatible =                                                          \
            utcmma_tmem_fits(M, instruction_m, N, static_cast<int>(sizeof(CPrecision)));                 \
        static constexpr bool valid = m_compatible and n_compatible and k_compatible and tmem_compatible; \
        using instruction                   = cute::conditional_t<valid,                                 \
                                                                  InstructionName<APrecision,            \
                                                                                  BPrecision,            \
                                                                                  CPrecision,            \
                                                                                  instruction_m,         \
                                                                                  instruction_n,         \
                                                                                  cute::UMMA::Major::K,  \
                                                                                  cute::UMMA::Major::K>, \
                                                                  void>;                                 \
    };

#if defined(CUBLASDX_CUTLASS_VERSION) && CUBLASDX_CUTLASS_VERSION >= 40501
#    define CUBLASDX_UTCMMA_FP8_INSTRUCTION(InstructionName, APrecision, BPrecision, CPrecision, M, N, IsAKMajor, IsBKMajor) \
        InstructionName<APrecision,                                                                                          \
                        BPrecision,                                                                                          \
                        CPrecision,                                                                                          \
                        M,                                                                                                   \
                        N,                                                                                                   \
                        IsAKMajor ? cute::UMMA::Major::K : cute::UMMA::Major::MN,                                           \
                        IsBKMajor ? cute::UMMA::Major::K : cute::UMMA::Major::MN,                                           \
                        cute::UMMA::ScaleIn::One,                                                                            \
                        cute::UMMA::ScaleIn::One>
#else
#    define CUBLASDX_UTCMMA_FP8_INSTRUCTION(InstructionName, APrecision, BPrecision, CPrecision, M, N, IsAKMajor, IsBKMajor)  \
        cute::MMA_Traits<InstructionName,                                                                                     \
                         APrecision,                                                                                         \
                         BPrecision,                                                                                         \
                         CPrecision,                                                                                         \
                         cute::C<M>,                                                                                         \
                         cute::C<N>,                                                                                         \
                         cute::integral_constant<cute::UMMA::Major, IsAKMajor ? cute::UMMA::Major::K : cute::UMMA::Major::MN>, \
                         cute::integral_constant<cute::UMMA::Major, IsBKMajor ? cute::UMMA::Major::K : cute::UMMA::Major::MN>, \
                         cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>,                              \
                         cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>>
#endif

#define UTCMMA_FP8_DEFINITION(APrecision, BPrecision, CPrecision, InstructionName)                                  \
    template<int M, int N, int K, bool IsAKMajor, bool IsBKMajor>                                                   \
    struct get_best_utcmma_instruction<M, N, K, APrecision, BPrecision, CPrecision, IsAKMajor, IsBKMajor> {         \
        static constexpr int  dummy_m       = 64;                                                                   \
        static constexpr int  dummy_n       = 64;                                                                   \
        static constexpr bool m_compatible  = (M % 128 == 0) or (M == 64);                                          \
        static constexpr int  instruction_m = m_compatible ? ((M % 128 == 0) ? 128 : 64) : dummy_m;                 \
        static constexpr bool n_compatible  = is_utcmma_fp8_n_compatible(N, IsBKMajor);                         \
        static constexpr int  instruction_n = n_compatible ? N : dummy_n;                                           \
        static constexpr int  instruction_k = 256 / cute::sizeof_bits<APrecision>::value;                           \
        static constexpr bool k_compatible  = K % instruction_k == 0;                                               \
        static constexpr bool tmem_compatible =                                                                     \
            utcmma_tmem_fits(M, instruction_m, N, static_cast<int>(sizeof(CPrecision)));                            \
        static constexpr bool valid = m_compatible and n_compatible and k_compatible and tmem_compatible;           \
        using instruction                   = cute::conditional_t<valid,                                             \
                                                CUBLASDX_UTCMMA_FP8_INSTRUCTION(InstructionName,                    \
                                                                                APrecision,                        \
                                                                                BPrecision,                        \
                                                                                CPrecision,                        \
                                                                                instruction_m,                     \
                                                                                instruction_n,                     \
                                                                                IsAKMajor,                         \
                                                                                IsBKMajor),                        \
                                                void>;                                                               \
    };

#define UTCMMA_S8_DEFINITION(APrecision, BPrecision, CPrecision, InstructionName)                           \
    template<int M, int N, int K, bool IsAKMajor, bool IsBKMajor>                                           \
    struct get_best_utcmma_instruction<M, N, K, APrecision, BPrecision, CPrecision, IsAKMajor, IsBKMajor> { \
        static constexpr int  dummy_m       = 64;                                                           \
        static constexpr int  dummy_n       = 64;                                                           \
        static constexpr bool m_compatible  = (M % 128 == 0) or (M == 64);                                  \
        static constexpr int  instruction_m = m_compatible ? ((M % 128 == 0) ? 128 : 64) : dummy_m;         \
        static constexpr bool n_compatible  = is_utcmma_s8_n_compatible(N);                                \
        static constexpr int  instruction_n = n_compatible ? N : dummy_n;                                   \
        static constexpr int  instruction_k = 256 / cute::sizeof_bits<APrecision>::value;                   \
        static constexpr bool k_compatible  = K % instruction_k == 0;                                       \
        static constexpr bool tmem_compatible =                                                             \
            utcmma_tmem_fits(M, instruction_m, N, static_cast<int>(sizeof(CPrecision)));                    \
        static constexpr bool valid = m_compatible and n_compatible and k_compatible and tmem_compatible;   \
        using instruction =                                                                                 \
            cute::conditional_t<valid,                                                                      \
                                InstructionName<APrecision,                                                 \
                                                BPrecision,                                                 \
                                                CPrecision,                                                 \
                                                instruction_m,                                              \
                                                instruction_n,                                              \
                                                IsAKMajor ? cute::UMMA::Major::K : cute::UMMA::Major::MN,   \
                                                IsBKMajor ? cute::UMMA::Major::K : cute::UMMA::Major::MN>,  \
                                void>;                                                                      \
    };

            // Explicit UTCMMA instances
            UTCMMA_FP8_DEFINITION(float_e4m3_t, float_e4m3_t, half_t, cute::SM100_MMA_F8F6F4_SS);
            UTCMMA_FP8_DEFINITION(float_e5m2_t, float_e4m3_t, half_t, cute::SM100_MMA_F8F6F4_SS);
            UTCMMA_FP8_DEFINITION(float_e4m3_t, float_e5m2_t, half_t, cute::SM100_MMA_F8F6F4_SS);
            UTCMMA_FP8_DEFINITION(float_e5m2_t, float_e5m2_t, half_t, cute::SM100_MMA_F8F6F4_SS);

            UTCMMA_FP8_DEFINITION(float_e4m3_t, float_e4m3_t, float, cute::SM100_MMA_F8F6F4_SS);
            UTCMMA_FP8_DEFINITION(float_e5m2_t, float_e4m3_t, float, cute::SM100_MMA_F8F6F4_SS);
            UTCMMA_FP8_DEFINITION(float_e4m3_t, float_e5m2_t, float, cute::SM100_MMA_F8F6F4_SS);
            UTCMMA_FP8_DEFINITION(float_e5m2_t, float_e5m2_t, float, cute::SM100_MMA_F8F6F4_SS);

            UTCMMA_TF32_DEFINITION(cutlass::tfloat32_t, cutlass::tfloat32_t, float, cute::SM100_MMA_TF32_SS);
            UTCMMA_S8_DEFINITION(int8_t, int8_t, int32_t, cute::SM100_MMA_S8_SS);
            UTCMMA_S8_DEFINITION(uint8_t, int8_t, int32_t, cute::SM100_MMA_S8_SS);
            UTCMMA_S8_DEFINITION(int8_t, uint8_t, int32_t, cute::SM100_MMA_S8_SS);
            UTCMMA_S8_DEFINITION(uint8_t, uint8_t, int32_t, cute::SM100_MMA_S8_SS);

            UTCMMA_DEFINITION(half_t, half_t, float, cute::SM100_MMA_F16BF16_SS);
            UTCMMA_DEFINITION(bfloat16_t, bfloat16_t, float, cute::SM100_MMA_F16BF16_SS);

            // UTCMMA --> Blackwell only MMA instructions
            // =====================================
            template<int SM,
                     int NumWarps,
                     class GEMMShape,
                     class TA,
                     class TB,
                     class TC,
                     bool IsAKMajor,
                     bool IsBKMajor>
            struct get_best_utcmma_tiled_instruction {
                static constexpr bool architecture_compatible =
                    (SM == 1000) or (SM == 1030) or (SM == 1100);
                // UTCMMA is always launched by 1 thread, tiling is in regard to SMs (as there exists a 2SM)
                using default_tile = cute::Layout<cute::_1, cute::_1>;

                static constexpr int m = cute::get<0>(GEMMShape {});
                static constexpr int n = cute::get<1>(GEMMShape {});
                static constexpr int k = cute::get<2>(GEMMShape {});

                using instruction =
                    typename get_best_utcmma_instruction<m, n, k, TA, TB, TC, IsAKMajor, IsBKMajor>::instruction;

                using tile = cute::conditional_t<cute::is_void_v<instruction>, void, default_tile>;
                static constexpr instruction_type instruction_kind = instruction_type::utcmma;
            };
        } // namespace layout_database
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_SUGGESTED_UTCMMA_HPP
