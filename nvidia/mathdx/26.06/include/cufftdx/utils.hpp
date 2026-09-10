// Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_UTILS_HPP
#define CUFFTDX_UTILS_HPP

#if !defined(__CUDACC_RTC__)
#    include <vector>
#endif

#if !defined(__CUDACC_RTC__)
#    include "cufftdx/detail/runtime_query_helpers.hpp"
#endif
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#    include "cufftdx/traits/fft_traits.hpp"
#endif
namespace cufftdx {
    namespace experimental {
        enum class query_code_type
        {
            ptx,
            ltoir_online,
            ltoir_offline
        };
    } // namespace experimental

    namespace utils {
        using detail::algorithm;
        using detail::execution_type;
    } // namespace utils

    namespace experimental {
        namespace utils {
            using detail::backend_impl_traits;
            using detail::frontend_impl_traits;
            using detail::frontend_to_backend;

            constexpr unsigned int get_shared_memory_size_for_dynamic_batching(const unsigned int shared_memory_size_per_fft, const unsigned int ffts_per_block, const unsigned int implicit_type_batching) {
                return (implicit_type_batching != 0) ? (shared_memory_size_per_fft * (ffts_per_block + implicit_type_batching - 1) / implicit_type_batching) : 0;
            }

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
            template<class FFT>
            constexpr unsigned int get_shared_memory_size_for_dynamic_batching(const unsigned int ffts_per_block) {
                static_assert(cufftdx::experimental::is_dynamic_batching_enabled_v<FFT>, "Dynamic batching is not enabled for this FFT");
                return get_shared_memory_size_for_dynamic_batching(FFT::shared_memory_size, ffts_per_block, FFT::implicit_type_batching);
            }
#endif

        } // namespace utils
    } // namespace experimental
} // namespace cufftdx
#if defined(CUFFTDX_ENABLE_CUFFT_DEPENDENCY) && !defined(__CUDACC_RTC__)
#    include "cufftdx/utils/cufft_lto.hpp"
#endif

// Unified public API that dispatches based on query_code_type
namespace cufftdx {
    namespace experimental {
        namespace utils {
#if defined(CUFFTDX_ENABLE_RUNTIME_DATABASE) && !defined(__CUDACC_RTC__)
            template<bool checkLTODatabaseDefined = true>
            inline std::optional<frontend_impl_traits> query_database(unsigned                                             fft_size,
                                                                       fft_direction                                        dir,
                                                                       fft_type                                             type,
                                                                       unsigned                                             sm,
                                                                       cufftdx::utils::execution_type                      execution,
                                                                       precision                                            prec      = precision::f32,
                                                                       unsigned                                             fft_ept   = 0,
                                                                       unsigned                                             fpb       = 0,
                                                                       std::tuple<unsigned int, unsigned int, unsigned int> block_dim = {0, 0, 0},
                                                                       complex_layout                                       layout    = complex_layout::natural,
                                                                       real_mode                                            rmode     = real_mode::normal,
                                                                       query_code_type                                      code_type = query_code_type::ptx) {
                switch (code_type) {
                    case query_code_type::ptx:
                        return cufftdx::detail::query_database_offline<checkLTODatabaseDefined>(
                            fft_size, dir, type, sm, execution, prec, fft_ept, fpb, block_dim, layout, rmode, cufftdx::experimental::code_type::ptx);
                    case query_code_type::ltoir_offline:
                        return cufftdx::detail::query_database_offline<checkLTODatabaseDefined>(
                            fft_size, dir, type, sm, execution, prec, fft_ept, fpb, block_dim, layout, rmode, cufftdx::experimental::code_type::ltoir);
                    case query_code_type::ltoir_online:
#    ifdef CUFFTDX_ENABLE_CUFFT_DEPENDENCY
                        return cufftdx::detail::query_single_cufft_implementation(
                            fft_size, dir, type, sm, execution, prec, layout, rmode, fft_ept, fpb, block_dim);
#    else
                        return std::nullopt;
#    endif
                }
                return std::nullopt;
            }

            template<bool checkLTODatabaseDefined = true>
            inline std::vector<frontend_impl_traits> get_all_implementations(unsigned                                             fft_size,
                                                                              fft_direction                                        dir,
                                                                              fft_type                                             type,
                                                                              unsigned                                             sm,
                                                                              cufftdx::utils::execution_type                      execution,
                                                                              precision                                            prec      = precision::f32,
                                                                              unsigned                                             fft_ept   = 0,
                                                                              unsigned                                             fpb       = 0,
                                                                              std::tuple<unsigned int, unsigned int, unsigned int> block_dim = {0, 0, 0},
                                                                              complex_layout                                       layout    = complex_layout::natural,
                                                                              real_mode                                            rmode     = real_mode::normal,
                                                                              query_code_type                                      code_type = query_code_type::ptx) {
                switch (code_type) {
                    case query_code_type::ptx:
                        return cufftdx::detail::get_all_implementations_offline<checkLTODatabaseDefined>(
                            fft_size, dir, type, sm, execution, prec, fft_ept, fpb, block_dim, layout, rmode, cufftdx::experimental::code_type::ptx);
                    case query_code_type::ltoir_offline:
                        return cufftdx::detail::get_all_implementations_offline<checkLTODatabaseDefined>(
                            fft_size, dir, type, sm, execution, prec, fft_ept, fpb, block_dim, layout, rmode, cufftdx::experimental::code_type::ltoir);
                    case query_code_type::ltoir_online:
#    ifdef CUFFTDX_ENABLE_CUFFT_DEPENDENCY
                        return cufftdx::utils::query_all_cufft_implementations(
                            fft_size, dir, type, sm, execution, prec, layout, rmode, fft_ept, fpb, block_dim);
#    else
                        return {};
#    endif
                }
                return {};
            }
#endif // CUFFTDX_ENABLE_RUNTIME_DATABASE
        } // namespace utils
    } // namespace experimental
} // namespace cufftdx

#endif // CUFFTDX_UTILS_HPP
