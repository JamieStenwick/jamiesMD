// Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_TENSOR_HPP
#define CUBLASDX_DETAIL_TENSOR_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "commondx/tensor.hpp"

#include "cublasdx/detail/cute/cute_tensor.hpp"
#include "cublasdx/detail/cute/extension/cute_compatibility.hpp"

namespace commondx {
    template<class Iter, class SwizzleFn, int B, class Layout>
    CUBLASDX_HOST_DEVICE auto make_tensor(
        Iter*                                                                       iter,
        cute::ComposedLayout<SwizzleFn, cute::smem_ptr_flag_bits<B>, Layout> const& layout) {
        if constexpr (sizeof(Iter) == (B / 8)) {
            return cute::make_tensor(cute::make_smem_ptr(iter), layout);
        } else {
            // Recast suggested layout for using with decoupled precision
            return cute::make_tensor(
                cute::make_smem_ptr(iter),
                cute::make_composed_layout(
                    layout.layout_a(), cute::smem_ptr_flag_bits<cute::sizeof_bits_v<Iter>> {}, layout.layout_b()));
        }
        CUTE_GCC_UNREACHABLE;
    }

    template<class Iter, class SwizzleFn, int B, class Layout>
    CUBLASDX_HOST_DEVICE auto make_tensor(
        cute::smem_ptr<Iter*> const&                                                iter,
        cute::ComposedLayout<SwizzleFn, cute::smem_ptr_flag_bits<B>, Layout> const& layout) {
        if constexpr (sizeof(Iter) == (B / 8)) {
            return cute::make_tensor(iter, layout);
        } else {
            // Recast suggested layout for using with decoupled precision
            return cute::make_tensor(
                iter,
                cute::make_composed_layout(
                    layout.layout_a(), cute::smem_ptr_flag_bits<cute::sizeof_bits_v<Iter>> {}, layout.layout_b()));
        }
        CUTE_GCC_UNREACHABLE;
    }

    template<class Iter, class Tag, class SwizzleFn, int B, class Layout>
    CUBLASDX_HOST_DEVICE auto make_tensor(
        Iter* iter,
        detail::pointer_layout<Tag, cute::ComposedLayout<SwizzleFn, cute::smem_ptr_flag_bits<B>, Layout>> const&
            layout) {
        if constexpr (sizeof(Iter) == (B / 8)) {
            return cute::make_tensor(cute::make_smem_ptr(iter), layout.layout);
        } else {
            // Recast suggested layout for using with decoupled precision
            return cute::make_tensor(cute::make_smem_ptr(iter),
                                     cute::make_composed_layout(layout.layout.layout_a(),
                                                                cute::smem_ptr_flag_bits<cute::sizeof_bits_v<Iter>> {},
                                                                layout.layout.layout_b()));
        }
        CUTE_GCC_UNREACHABLE;
    }

    template<class Functor, class Iter, class Layout>
    CUBLASDX_HOST_DEVICE auto make_tensor(cute::transform_iter<Functor, Iter> const& iter,
                                          Layout const& layout) {
        return cute::make_tensor(iter, layout);
    }

    template<class Functor, class Iter, class Layout>
    CUBLASDX_HOST_DEVICE auto make_tensor(cute::ViewEngine<cute::transform_iter<Functor, Iter>> const& iter,
                                          Layout const& layout) {
        return cute::make_tensor(iter, layout);
    }

    template<class Iter, class Tag, class SwizzleFn, int B, class Layout>
    CUBLASDX_HOST_DEVICE auto make_tensor(
        cute::smem_ptr<Iter*> const& iter,
        detail::pointer_layout<Tag, cute::ComposedLayout<SwizzleFn, cute::smem_ptr_flag_bits<B>, Layout>> const&
            layout) {
        if constexpr (sizeof(Iter) == (B / 8)) {
            return cute::make_tensor(iter, layout.layout);
        } else {
            // Recast suggested layout for using with decoupled precision
            return cute::make_tensor(iter,
                                     cute::make_composed_layout(layout.layout.layout_a(),
                                                                cute::smem_ptr_flag_bits<cute::sizeof_bits_v<Iter>> {},
                                                                layout.layout.layout_b()));
        }
        CUTE_GCC_UNREACHABLE;
    }
} // namespace commondx

namespace cublasdx {
    using commondx::cosize;
    using commondx::is_layout;
    using commondx::is_layout_v;
    using commondx::make_tensor;
    using commondx::size;
    using commondx::tensor;

    CUTE_INLINE_CONSTANT auto slice = cute::_;

    template<class Tiler>
    CUBLASDX_DEVICE auto shape(Tiler const& tiler) {
        return cute::shape(tiler);
    }

    template<class Tag, class Layout>
    CUBLASDX_DEVICE auto shape(commondx::detail::pointer_layout<Tag, Layout> const& tiler) {
        return cute::shape(tiler.layout);
    }

    // Regular casting
    template<class NewType, class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<Engine, Layout>& tensor) {
        return cute::recast<NewType>(tensor);
    }

    template<class NewType, class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<Engine, Layout> const& tensor) {
        return cute::recast<NewType>(tensor);
    }

    template<class NewType, class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<Engine, Layout>&& tensor) {
        return safe_recast<NewType>(tensor);
    }

    // transform_iter
    template<class NewType, class Functor,class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<cute::transform_iter<Functor, Engine>, Layout>& tensor) {
        return cute::lazy::transform(tensor, [](auto value) -> NewType { return reinterpret_cast<NewType&>(value); });
    }

    template<class NewType, class Functor, class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<cute::transform_iter<Functor, Engine>, Layout> const& tensor) {
        return cute::lazy::transform(tensor, [](auto value) -> NewType { return reinterpret_cast<NewType const&>(value); });
    }

    template<class NewType,class Functor, class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<cute::transform_iter<Functor, Engine>, Layout>&& tensor) {
        return safe_recast<NewType>(tensor);
    }

    // ViewEngine<transform_iter>
    template<class NewType, class Functor,class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<cute::ViewEngine<cute::transform_iter<Functor, Engine>>, Layout>& tensor) {
        return cute::lazy::transform(tensor, [](auto value) -> NewType { return reinterpret_cast<NewType&>(value); });
    }

    template<class NewType, class Functor, class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<cute::ViewEngine<cute::transform_iter<Functor, Engine>>, Layout> const& tensor) {
        return cute::lazy::transform(tensor, [](auto value) -> NewType { return reinterpret_cast<NewType const&>(value); });
    }

    template<class NewType,class Functor, class Engine, class Layout>
    CUBLASDX_HOST_DEVICE auto safe_recast(cute::Tensor<cute::ViewEngine<cute::transform_iter<Functor, Engine>>, Layout>&& tensor) {
        return safe_recast<NewType>(tensor);
    }

    template<class Tensor, class Tiler>
    CUBLASDX_DEVICE auto get_tile_row(Tensor&& tensor, Tiler tiler, unsigned tile_row) {
        static_assert(cute::rank_v<Tensor> == 2 and cute::rank_v<Tiler> == 2,
                      "Both tensor and tiler must be 2D tensors");
        auto tiler_shape = cute::make_shape(cute::size<0>(tiler), cute::size<1>(tiler));
        static_assert(cute::is_static<decltype(tiler_shape)>::value, "Tiler must have static shape");
        return cute::local_tile(tensor, tiler_shape, cute::make_coord(tile_row, cublasdx::slice));
    }

    template<class Tensor, class Tiler>
    CUBLASDX_DEVICE auto get_tile_col(Tensor&& tensor, Tiler tiler, unsigned tile_col) {
        static_assert(cute::rank_v<Tensor> == 2 and cute::rank_v<Tiler> == 2,
                      "Both tensor and tiler must be 2D tensors");
        auto tiler_shape = cute::make_shape(cute::size<0>(tiler), cute::size<1>(tiler));
        static_assert(cute::is_static<decltype(tiler_shape)>::value, "Tiler must have static shape");
        return cute::local_tile(tensor, tiler_shape, cute::make_coord(cublasdx::slice, tile_col));
    }

    template<class Tensor, class Tiler>
    CUBLASDX_DEVICE auto get_tile(Tensor&& tensor, Tiler tiler, unsigned tile_row, unsigned tile_col) {
        static_assert(cute::rank_v<Tensor> == 2 and cute::rank_v<Tiler> == 2,
                      "Both tensor and tiler must be 2D tensors");
        auto tiler_shape = cute::make_shape(cute::size<0>(tiler), cute::size<1>(tiler));
        static_assert(cute::is_static<decltype(tiler_shape)>::value, "Tiler must have static shape");
        return cute::local_tile(tensor, tiler_shape, cute::make_coord(tile_row, tile_col));
    }

    template<class BLAS, class ATensor, class BTensor, class CTensor>
    CUBLASDX_DEVICE auto get_tiles(ATensor&& tensor_a,
                                   BTensor&& tensor_b,
                                   CTensor&& tensor_c,
                                   unsigned  tile_row,
                                   unsigned  tile_col) {
        const auto tile_row_a = get_tile_row(tensor_a, BLAS::a_shape, tile_row);
        const auto tile_col_b = get_tile_col(tensor_b, BLAS::b_shape, tile_col);
        const auto tile_c     = get_tile(tensor_c, BLAS::c_shape, tile_row, tile_col);
        return cute::make_tuple(tile_row_a, tile_col_b, tile_c);
    }

    template<arrangement Arrangement, typename T>
    CUBLASDX_HOST_DEVICE auto make_gmem_tensor(T ptr, unsigned const x, unsigned const y) {
        const auto stride_x = cute::conditional_return<Arrangement == cublasdx::col_major>(cute::_1 {}, y);
        const auto stride_y = cute::conditional_return<Arrangement == cublasdx::row_major>(cute::_1 {}, x);
        return cute::make_tensor(cute::make_gmem_ptr(ptr),
                                 cute::make_layout(cute::make_shape(x, y), cute::make_stride(stride_x, stride_y)));
    }

    template<arrangement Arrangement, typename T>
    CUBLASDX_HOST_DEVICE auto make_gmem_tensor(T ptr, unsigned const x, unsigned const y, unsigned const ld) {
        const auto stride_x = cute::conditional_return<Arrangement == cublasdx::col_major>(cute::_1 {}, ld);
        const auto stride_y = cute::conditional_return<Arrangement == cublasdx::row_major>(cute::_1 {}, ld);
        return cute::make_tensor(cute::make_gmem_ptr(ptr),
                                 cute::make_layout(cute::make_shape(x, y), cute::make_stride(stride_x, stride_y)));
    }

    // Creates a 3D global-memory tensor (x x y x batches) with contiguous per-batch
    // stride x*y and arrangement-aware inner strides.  Useful for TRSM where each
    // thread block processes BPB consecutive batches stored back-to-back in memory.
    template<arrangement Arrangement, typename T>
    CUBLASDX_HOST_DEVICE auto make_gmem_tensor_batched(T ptr, unsigned const x, unsigned const y, unsigned const batches) {
        const auto stride_x = cute::conditional_return<Arrangement == cublasdx::col_major>(cute::_1 {}, y);
        const auto stride_y = cute::conditional_return<Arrangement == cublasdx::col_major>(x, cute::_1 {});
        return cute::make_tensor(cute::make_gmem_ptr(ptr),
                                 cute::make_layout(cute::make_shape(x, y, batches),
                                                   cute::make_stride(stride_x, stride_y, x * y)));
    }

    // Extracts the per-block tile from a 3D global-memory tensor
    // (rows x cols x total_batches) using a BLAS gmem layout to drive the tiling:
    //   layout is 2D (BPB == 1): returns a 2D tensor (rows x cols) - a single batch slice.
    //   layout is 3D (BPB  > 1): returns a 3D tensor (rows x cols x BPB) - a multi-batch tile.
    // The returned rank matches what BLAS::get_layout_smem_a/b produce for the same BPB,
    // so copies between the block tile and smem tensors are always rank-compatible.
    // Usage: cublasdx::get_batch(global_a, BLAS::get_layout_gmem_a(), blockIdx.x)
    template<class Tensor, class TilerShape>
    CUBLASDX_HOST_DEVICE auto get_batch(Tensor&&                                              tensor,
                                        TilerShape const&                                     tiler_shape,
                                        unsigned const                                        block_idx) {
        static_assert(cute::rank_v<cute::remove_cvref_t<Tensor>> == 3,
                      "get_batch requires a 3D tensor (rows x cols x total_batches)");
        return cute::local_tile(static_cast<Tensor&&>(tensor),
                                cute::shape(cublasdx::detail::get_cute_layout(tiler_shape)),
                                cute::make_coord(cute::Int<0> {}, cute::Int<0> {}, block_idx));
    }

    template<class BEngine, class BLayout>
    CUBLASDX_HOST_DEVICE auto transpose_view(const tensor<BEngine, BLayout>& tensor_b) {
        if constexpr(decltype(cute::rank(tensor_b))::value == 1) {
            return tensor_b;
        } else if constexpr(decltype(cute::rank(tensor_b))::value <= 3) {
            return cublasdx::detail::cute_backend::transpose_first_two_modes_of_tensor(tensor_b);
        } else {
            static_assert(decltype(cute::rank(tensor_b))::value <= 3, "Transpose is only supported for 2D and 1D tensors");
        }
    }

    template<class BEngine, class BLayout>
    CUBLASDX_HOST_DEVICE auto transpose_view(tensor<BEngine, BLayout>& tensor_b) {
        if constexpr (decltype(cute::rank(tensor_b))::value == 1) {
            return tensor_b;
        } else if constexpr (decltype(cute::rank(tensor_b))::value <= 3) {
            return cublasdx::detail::cute_backend::transpose_first_two_modes_of_tensor(tensor_b);
        } else {
            static_assert(decltype(cute::rank(tensor_b))::value <= 3,
                          "Transpose is only supported for 2D and 1D tensors");
        }
    }

    template<class BEngine, class BLayout>
    CUBLASDX_HOST_DEVICE auto transpose_view(tensor<BEngine, BLayout>&& tensor_b) {
        return transpose_view(tensor_b);
    }

    template<class Tensor>
    CUBLASDX_HOST_DEVICE auto conjugate_view(Tensor&& tensor) {
        return cute::lazy::transform(tensor, conjugate {});
    }

    template<class Tensor>
    CUBLASDX_HOST_DEVICE auto conj_transpose_view(Tensor&& tensor) {
        return conjugate_view(transpose_view(tensor));
    }

    template<class Tensor, class Transform>
    CUBLASDX_HOST_DEVICE auto make_transform_view(Tensor&& tensor, Transform transform) {
        return cute::lazy::transform(tensor, transform);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_TENSOR_HPP
