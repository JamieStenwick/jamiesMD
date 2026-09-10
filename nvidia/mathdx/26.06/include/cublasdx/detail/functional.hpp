// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_FUNCTIONAL_HPP
#define CUBLASDX_DETAIL_FUNCTIONAL_HPP

#include "cublasdx/detail/cute/cute_tensor.hpp"

namespace cublasdx {
    struct conjugate {
        // For non-complex types (ie. real type) conjugate does nothing
        template<class T>
        CUBLASDX_HOST_DEVICE T operator()(T value) const {
            return cutlass::conj(value);
        }

        template<class T>
        CUBLASDX_HOST_DEVICE complex<T> operator()(complex<T> value) const {
            // NOTE: Negating int8_t by unary (-) operator return int32_t,
            // this static_cast is necessary to avoid narrowing conversion
            // warnings
            using cublasdx::detail::cast_from_cutlass_type;
            using cublasdx::detail::cast_to_cutlass_type;

            // Since FP8 types do not have builtin operators, casting to CUTLASS
            // types will handle it out of the box
            using ct = cublasdx::detail::convert_to_cutlass_type_t<T>;
            // This explicit cast is required because negate(int8_t) -> int32_t
            const auto negated = cast_from_cutlass_type<T>(static_cast<ct>(-cast_to_cutlass_type<ct>(value.imag())));
            return complex<T> {static_cast<T>(value.real()), negated};
        }
    };


    struct identity {
        template<class Arg>
        CUBLASDX_HOST_DEVICE Arg operator()(Arg const& arg) const {
            return arg;
        }

        CUBLASDX_HOST_DEVICE
        void operator()() const {}
    };

    namespace detail {

        template<bool IsInvocable, class Functor, class... Args>
        struct invoke_result_impl {
            using type = void;
        };

        template<class Functor, class... Args>
        struct invoke_result_impl<true, Functor, Args...> {
            using type = COMMONDX_STL_NAMESPACE::invoke_result_t<Functor, Args...>;
        };

        template<class Functor, class... Args>
        struct invoke_result:
            invoke_result_impl<COMMONDX_STL_NAMESPACE::is_invocable_v<Functor, Args...>, Functor, Args...> {};

        template<class Functor, class... Args>
        using res_t = COMMONDX_STL_NAMESPACE::decay_t<typename invoke_result<Functor, Args...>::type>;

        template<typename MemOp, typename Input, typename Output>
        CUBLASDX_HOST_DEVICE constexpr bool is_functor_compatible() {
            using invoke_result =
                COMMONDX_STL_NAMESPACE::decay_t<COMMONDX_STL_NAMESPACE::invoke_result_t<MemOp, Input>>;
            return (((sizeof(invoke_result) == sizeof(Output)) && (alignof(invoke_result) == alignof(Output))) ||
                    cute::is_convertible_v<invoke_result, Output>);
        }

        template<class F1, class F2>
        struct composed_functor {
            template<class T>
            CUBLASDX_DEVICE auto operator()(T value) const {
                return F2()(F1()(value));
            }
        };

        template<class F1, class F2>
        CUBLASDX_HOST_DEVICE auto compose_functors(const F1& f1, const F2& f2) {
            static_assert(cute::is_static<F1>::value, "F1 functor must be stateless");
            static_assert(cute::is_static<F2>::value, "F2 functor must be stateless");

            if constexpr (cute::is_same_v<F1, identity> && cute::is_same_v<F2, identity>) {
                return f1;
            } else if constexpr (cute::is_same_v<F1, identity>) {
                return f2;
            } else if constexpr (cute::is_same_v<F2, identity>) {
                return f1;
            } else {
                return composed_functor<F1, F2> {};
            }
        }

        template<class trans_op, class default_input_type>
        struct transform_op_wrapper {
            trans_op op;
            static_assert(cute::is_static<trans_op>::value, "Transformation functors must be stateless");

            using result_t =
                COMMONDX_STL_NAMESPACE::decay_t<typename invoke_result<trans_op, default_input_type>::type>;
            using cutlass_result_t = convert_to_cutlass_type_t<result_t>;

            CUBLASDX_HOST_DEVICE
            auto default_call(default_input_type const arg) const {
                auto result = op(arg);
                return cast_to_cutlass_type<cutlass_result_t>(result);
            }

            template<typename T>
            CUBLASDX_HOST_DEVICE auto operator()(const T& arg) const {
                // Is invokable with CUTLASS type
                if constexpr (COMMONDX_STL_NAMESPACE::is_invocable_v<trans_op, T>) {
                    using cutlass_input_result_t =
                        COMMONDX_STL_NAMESPACE::decay_t<typename invoke_result<trans_op, T>::type>;
                    using cutlass_result_t = convert_to_cutlass_type_t<cutlass_input_result_t>;
                    if constexpr (COMMONDX_STL_NAMESPACE::is_convertible_v<cutlass_input_result_t, cutlass_result_t>) {
                        return static_cast<cutlass_result_t>(op(arg));
                    } else {
                        return default_call(cast_from_cutlass_type<default_input_type>(arg));
                    }
                } else {
                    return default_call(cast_from_cutlass_type<default_input_type>(arg));
                }
            }
        };

        template<class T>
        struct is_identity_op: cute::false_type {
        };

        template<>
        struct is_identity_op<cublasdx::identity>: cute::true_type {
        };

        template<class Out>
        struct is_identity_op<transform_op_wrapper<cublasdx::identity, Out>>: cute::true_type {
        };

        template<class T>
        inline constexpr bool is_identity_op_v = is_identity_op<T>::value;
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_FUNCTIONAL_HPP
