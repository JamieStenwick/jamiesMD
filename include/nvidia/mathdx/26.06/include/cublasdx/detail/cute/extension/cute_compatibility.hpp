// Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_CUTE_COMPATIBILITY_UTILS_HPP
#define CUBLASDX_CUTE_COMPATIBILITY_UTILS_HPP

#include "cublasdx/detail/cute/cute_tensor.hpp"
#include "cublasdx/detail/functional.hpp"

namespace cublasdx {
    namespace detail {

        template<class Layout>
        CUBLASDX_HOST_DEVICE constexpr auto get_cute_layout(Layout const& layout) {
            if constexpr (cute::is_layout<Layout>::value or cute::is_tuple<Layout>::value) {
                return layout;
            } else {
                // Assume commondx::pointer_layout
                return layout.layout;
            }
        }

        template<class Layout>
        CUBLASDX_HOST_DEVICE constexpr auto 
        remove_zeros_from_layout(Layout const& layout) {
            if constexpr (cute::is_composed_layout<Layout>::value) {
                return cute::make_composed_layout(layout.layout_a(), 
                                                  layout.offset(), 
                                                  cute::make_layout(layout.layout_b().shape(),
                                                                    cute::filter_zeros(layout.layout_b().stride())));
            } else {
                return cute::filter_zeros(layout);
            }
        }
        
        namespace cute_backend {
            // CuTe backend details
            using cute::Int;
            using cute::Layout;
            using cute::Shape;
            using cute::Tensor;

            template<cublasdx::arrangement A, typename D1, typename D2, typename LD>
            constexpr auto make_layout_from_arrangement(D1, D2, LD ld) {
                static_assert(cute::is_integral<D1>::value && cute::is_integral<D2>::value &&
                              cute::is_integral<LD>::value);
                static_assert(cute::is_static_v<D1> && cute::is_static_v<D2>);

                constexpr bool is_rm = (A == cublasdx::row_major);

                using shape_t = Shape<D1, D2>;

                if constexpr (cute::is_static_v<LD>) {
                    constexpr auto ldval = cute::get<0>(ld);
                    using stride_t       = cute::Stride<Int<is_rm ? ldval : 1>, Int<is_rm ? 1 : ldval>>;

                    return Layout<shape_t, stride_t> {};
                } else {
                    const auto stride_1 = cute::conditional_return<is_rm>(ld, cute::_1 {});
                    const auto stride_2 = cute::conditional_return<is_rm>(cute::_1 {}, ld);
                    const auto stride   = cute::make_stride(stride_1, stride_2);
                    return cute::make_layout(shape_t {}, stride);
                }
            }

            template<class Layout, int ... Indices>
            constexpr auto transpose_select_dispatch(const Layout& l, cute::int_sequence<0, 1, Indices...>) {
                return cute::select<1, 0, Indices...>(l);
            }

            template<class Layout>
            constexpr CUBLASDX_HOST_DEVICE auto transpose_first_two_modes_of_layout(const Layout& l) {
                static_assert(decltype(cute::rank(Layout {}))::value > 1, "Layout rank for transpose must be greater than 1");
                return transpose_select_dispatch(l, cute::make_int_sequence<decltype(cute::rank(Layout {}))::value> {});
            }

            template<typename Swizzle, typename Offset, typename Layout>
            constexpr CUBLASDX_HOST_DEVICE auto transpose_first_two_modes_of_layout(
                const cute::ComposedLayout<Swizzle, Offset, Layout>& l) {
                return cute::composition(l.layout_a(), l.offset(), transpose_first_two_modes_of_layout(l.layout_b()));
            }

            template<typename T, typename Layout>
            constexpr CUBLASDX_HOST_DEVICE auto transpose_first_two_modes_of_tensor(const cute::Tensor<T, Layout>& t) {
                return cute::make_tensor(t.data(), transpose_first_two_modes_of_layout(t.layout()));
            }

            template<typename T, typename Layout>
            constexpr CUBLASDX_HOST_DEVICE auto transpose_first_two_modes_of_tensor(cute::Tensor<T, Layout>& t) {
                return cute::make_tensor(t.data(), transpose_first_two_modes_of_layout(t.layout()));
            }

            template<typename T, typename Layout>
            constexpr CUBLASDX_HOST_DEVICE auto transpose_first_two_modes_of_tensor(cute::Tensor<T, Layout>&& t) {
                return transpose_first_two_modes_of_tensor(t);
            }

            template<transpose_mode TM>
            constexpr auto get_load_op_from_transpose() {
                if constexpr (TM == transpose_mode::conj_transposed) {
                    return conjugate {};
                } else {
                    return identity {};
                }
            }
        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_CUTE_COMPATIBILITY_UTILS_HPP
