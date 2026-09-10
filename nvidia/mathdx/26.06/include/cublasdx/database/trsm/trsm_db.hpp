#ifndef CUBLASDX_DATABASE_TRSM_DB_HPP
#define CUBLASDX_DATABASE_TRSM_DB_HPP

#include <cute/container/array.hpp>
#include <cublasdx/types.hpp>

namespace cublasdx::detail::trsm {

    // Size thresholds for suggesting batches per block
    template<class T, unsigned Arch>
    constexpr inline __device__ __host__ cute::array<unsigned, 5> suggested_bpb_size_thresholds() {
        if (cute::is_same_v<T, float>) {
            return {3, 3, 7, 11, 11};
        } else if (cute::is_same_v<T, double>) {
            return {3, 3, 4, 4, 8};
        } else if (cute::is_same_v<T, cublasdx::complex<float>>) {
            return {3, 3, 4, 4, 8};
        } else { // cute::is_same_v<T, cublasdx::complex<double>>
            return {3, 3, 3, 7, 11};
        }
    }

    template<class T, unsigned M, unsigned N, bool is_left, int Arch>
    constexpr inline __device__ __host__ unsigned suggested_batches() {
        constexpr auto     thresholds = suggested_bpb_size_thresholds<T, Arch>();
        constexpr unsigned impl_M     = is_left ? M : N;
        constexpr unsigned impl_N     = is_left ? N : M;

        // too large N could risk of running out of shared memory with high bpb
        if (impl_N > 64) {
            return 1;
        } else {
            if (impl_M <= thresholds[0]) {
                return 32;
            } else if (impl_M <= thresholds[1]) {
                return 16;
            } else if (impl_M <= thresholds[2]) {
                return 8;
            } else if (impl_M <= thresholds[3]) {
                return 4;
            } else if (impl_M <= thresholds[4]) {
                return 2;
            } else {
                return 1;
            }
        }
    }

    // Size thresholds for suggesting the block dim for a single batch
    template<class T, unsigned Arch>
    constexpr inline __device__ __host__ cute::array<unsigned, 4> suggested_block_dim_size_thresholds() {
        // Tuned for H100. For float and double real, tune until 128, and complex<float> and complex<double> until 96
        constexpr unsigned int INF = unsigned(-1);
        if (cute::is_same_v<T, float>) {
            return {32, 67, 128, INF};
        } else if (cute::is_same_v<T, double>) {
            return {32, 48, 128, INF};
        } else if (cute::is_same_v<T, cublasdx::complex<float>>) {
            return {31, 63, 96, INF};
        } else { // cute::is_same_v<T, cublasdx::complex<double>>
            return {31, 63, 96, INF};
        }
    }

    template<class T, unsigned M, unsigned BPB, int Arch>
    constexpr inline __device__ __host__ dim3 suggested_block_dim() {
        // Targets throughput bound cases

        if constexpr (BPB > 1) {
            if (M <= 28) {
                return BPB <= 32 ? 32 : 64;
            } else if (M <= (sizeof(T) <= 4 ? 64 : 32)) {
                if (BPB == 2) {
                    return 64;
                } else {
                    // Use 3 or 4 warps, depending on what results in fewer idle warps for the last wave of batches
                    unsigned rem_3 = (BPB % 3 == 0) ? 0 : 3 - (BPB % 3);
                    unsigned rem_4 = (BPB % 4 == 0) ? 0 : 4 - (BPB % 4);
                    return (rem_3 < rem_4) ? 96 : 128;
                }
            } else {
                // For large sizes, just use suggestion for 1 batch per block.
                return suggested_block_dim<T, M, 1, Arch>();
            }
        } else {
            constexpr auto thresholds = suggested_block_dim_size_thresholds<T, Arch>();
            if (M <= thresholds[0]) {
                return 32;
            } else if (M <= thresholds[1]) {
                return 64;
            } else if (M <= thresholds[2]) {
                return 128;
            } else if (M <= thresholds[3]) {
                return 256;
            } else {
                return 512;
            }
        }
    }

} // namespace cublasdx::detail::trsm

#endif // CUBLASDX_DATABASE_TRSM_DB_HPP
