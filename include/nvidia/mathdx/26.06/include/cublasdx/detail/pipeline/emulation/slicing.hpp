// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_PIPELINE_EMULATION_SLICING_HPP
#define CUBLASDX_DETAIL_PIPELINE_EMULATION_SLICING_HPP

#include "cublasdx/detail/commondx_config.hpp"
#include "cublasdx/detail/pipeline/emulation/exponents.hpp"
#include "cublasdx/detail/tensor.hpp"
#include "cublasdx/traits.hpp"
#include "cute/int_tuple.hpp"

#ifdef COMMONDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/cstdint>
#    include <cuda/std/type_traits>
#else
#    include <cstdint>
#    include <type_traits>
#endif

namespace cublasdx {
    namespace detail {

        enum class emulation_slice_matrix
        {
            a,
            b
        };

        template<class SliceValueType, unsigned NSlices>
        struct emulation_slice_array {
            SliceValueType values[NSlices] = {};

            template<class Index>
            CUBLASDX_HOST_DEVICE constexpr SliceValueType& operator[](Index const index) {
                return values[index];
            }

            template<class Index>
            CUBLASDX_HOST_DEVICE constexpr SliceValueType const& operator[](Index const index) const {
                return values[index];
            }
        };

        template<class SliceValueType, unsigned NSlices, class InputValueType>
        CUBLASDX_HOST_DEVICE emulation_slice_array<SliceValueType, NSlices>
        emulation_slices_from_fp(InputValueType const val, COMMONDX_STL_NAMESPACE::int32_t const exponent_shift) {
            static_assert(COMMONDX_STL_NAMESPACE::is_integral_v<SliceValueType>);
            static_assert(COMMONDX_STL_NAMESPACE::is_signed_v<SliceValueType>);
            static_assert(COMMONDX_STL_NAMESPACE::is_same_v<InputValueType, float> ||
                          COMMONDX_STL_NAMESPACE::is_same_v<InputValueType, double>);

            emulation_slice_array<SliceValueType, NSlices> slices = {};

            static constexpr double normalization_factor = 0x1.0p52;
            int skip_slices = 0;
            COMMONDX_STL_NAMESPACE::int64_t r = 0;
            COMMONDX_STL_NAMESPACE::uint8_t reg_pack = 0;

            emulation_fp64_structure r0 = {static_cast<double>(val)};
            int denorm_compensation = 0;
            if (r0.s.exponent == 0) {
                if (r0.d == 0.0) {
                    // Zero contributes no slices. Returning here avoids treating the
                    // biased zero exponent as a real value when padded tails are sliced.
                    return slices;
                } else {
                    r0.d = (r0.d * normalization_factor);
                    denorm_compensation = -52;
                }
            }

            int exp = r0.s.exponent + exponent_shift + denorm_compensation - emulation_fp64_bias;
            exp += (NSlices - 1) * emulation_slice_width<COMMONDX_STL_NAMESPACE::uint8_t>();

            int extra_width = (exp + 1) - 63;
            extra_width = extra_width > 0 ? extra_width : 0;
            skip_slices = cute::ceil_div(extra_width, emulation_slice_width<COMMONDX_STL_NAMESPACE::uint8_t>());
            exp -= skip_slices * emulation_slice_width<COMMONDX_STL_NAMESPACE::uint8_t>();

            if (exp < 0) {
                r = 0;
            } else {
                r0.s.exponent = static_cast<unsigned int>(exp + emulation_fp64_bias);
                r = static_cast<COMMONDX_STL_NAMESPACE::int64_t>(r0.d);
            }

            for (COMMONDX_STL_NAMESPACE::int64_t idx = 0;
                 idx < static_cast<COMMONDX_STL_NAMESPACE::int64_t>(NSlices);
                 idx++) {
                COMMONDX_STL_NAMESPACE::int64_t slice_idx = NSlices - 1 - idx;
                if (idx < skip_slices) {
                    reg_pack = 0;
                } else {
                    reg_pack = static_cast<COMMONDX_STL_NAMESPACE::uint8_t>(r);
                    slices[slice_idx] = static_cast<COMMONDX_STL_NAMESPACE::int8_t>(reg_pack);
                    r = (r >> emulation_slice_width<COMMONDX_STL_NAMESPACE::uint8_t>()) +
                        (reg_pack >> emulation_slice_width<COMMONDX_STL_NAMESPACE::int8_t>());
                }
            }

            return slices;
        }

        template<int BlockSize, emulation_slice_matrix SliceMatrix, class InTensor, class OutTensor>
        __launch_bounds__(BlockSize, 2) __global__ void emulation_max_reduce_kernel(InTensor const in_tensor,
                                                                                    OutTensor out_tensor) {
            static_assert(BlockSize > 0 && (BlockSize & (BlockSize - 1)) == 0,
                          "emulation_max_reduce_kernel requires power-of-two BlockSize");
            __shared__ double reduction[BlockSize];

            auto tid = threadIdx.x;
            auto bid = blockIdx.x;
            auto const shape = in_tensor.layout().shape();
            auto const reduction_size = (SliceMatrix == emulation_slice_matrix::a) ? cute::get<1>(shape) : cute::get<0>(shape);

            double local_max = 0.0;
            for (auto i = tid; i < reduction_size; i += BlockSize) {
                double value = 0.0;
                if constexpr (SliceMatrix == emulation_slice_matrix::a) {
                    value = static_cast<double>(in_tensor(bid, i));
                } else {
                    value = static_cast<double>(in_tensor(i, bid));
                }
                value = value < 0.0 ? -value : value;
                local_max = local_max > value ? local_max : value;
            }

            reduction[tid] = local_max;
            __syncthreads();

            for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    double other = reduction[tid + stride];
                    reduction[tid] = reduction[tid] > other ? reduction[tid] : other;
                }
                __syncthreads();
            }

            if (tid == 0) {
                out_tensor(bid) = emulation_max_to_exponent_shift(reduction[0]);
            }
        }

        template<int BlockSize, int Slices, emulation_slice_matrix SliceMatrix, class InTensor, class ShiftTensor, class OutTensor>
        __launch_bounds__(BlockSize, 2) __global__ void emulation_slice_kernel(InTensor const in_tensor,
                                                                               ShiftTensor const shift_tensor,
                                                                               OutTensor out_tensor,
                                                                               COMMONDX_STL_NAMESPACE::uint64_t const reduction_dim_size,
                                                                               COMMONDX_STL_NAMESPACE::uint64_t const total_size) {
            using in_datatype = cute::remove_cvref_t<typename InTensor::value_type>;
            using out_datatype = typename OutTensor::value_type;
            COMMONDX_STL_NAMESPACE::uint64_t const tid =
                static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(threadIdx.x) +
                static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(blockIdx.x) * BlockSize;
            if (tid >= total_size) {
                return;
            }

            auto slow_idx = tid / reduction_dim_size;
            auto fast_idx = tid % reduction_dim_size;

            auto const row_idx = (SliceMatrix == emulation_slice_matrix::a) ? slow_idx : fast_idx;
            auto const col_idx = (SliceMatrix == emulation_slice_matrix::a) ? fast_idx : slow_idx;
            const auto slices = emulation_slices_from_fp<out_datatype, Slices>(
                static_cast<in_datatype>(in_tensor(row_idx, col_idx)),
                shift_tensor(slow_idx));

            CUTE_UNROLL
            for (int elem = 0; elem < Slices; ++elem) {
                out_tensor(row_idx, col_idx, elem) = slices[elem];
            }
        }

        template<cublasdx::arrangement Arrangement, int Slices, typename T>
        CUBLASDX_HOST_DEVICE auto make_emulation_slice_tensor(T* ptr, unsigned const x, unsigned const y) {
            auto const stride_x = cute::conditional_return<Arrangement == cublasdx::col_major>(cute::_1 {}, y);
            auto const stride_y = cute::conditional_return<Arrangement == cublasdx::col_major>(x, cute::_1 {});
            auto const slice_stride = static_cast<COMMONDX_STL_NAMESPACE::uint64_t>(x) * y;
            auto const layout = cute::make_layout(cute::make_shape(x, y, cute::Int<Slices> {}),
                                                  cute::make_stride(stride_x, stride_y, slice_stride));
            return cute::make_tensor(cute::make_gmem_ptr(ptr), layout);
        }

        template<unsigned TileExtent, typename T>
        CUBLASDX_HOST_DEVICE auto make_emulation_shift_tensor(T* ptr, unsigned const tiles) {
            auto const layout = cute::make_layout(cute::make_shape(cute::Int<TileExtent> {}, tiles),
                                                  cute::make_stride(cute::_1 {}, cute::Int<TileExtent> {}));
            return cute::make_tensor(cute::make_gmem_ptr(ptr), layout);
        }

    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_PIPELINE_EMULATION_SLICING_HPP
