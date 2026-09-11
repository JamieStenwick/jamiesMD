// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_RUNTIME_QUERY_HELPERS_HPP
#define CUFFTDX_DETAIL_RUNTIME_QUERY_HELPERS_HPP

#include <optional>
#include <tuple>
#include <vector>

#include "cufftdx/database/detail/database_query.hpp"
#include "cufftdx/traits/detail/frontend_backend_mappings.hpp"

namespace cufftdx {
    namespace detail {

#if defined CUFFTDX_ENABLE_RUNTIME_DATABASE
        template<bool checkLTODatabaseDefined = true>
        std::optional<frontend_impl_traits> query_database_offline(unsigned                                             fft_size,
                                                                   fft_direction                                        dir,
                                                                   fft_type                                             type,
                                                                   unsigned                                             sm,
                                                                   execution_type                                       execution,
                                                                   precision                                            prec      = precision::f32,
                                                                   unsigned                                             fft_ept   = 0,
                                                                   unsigned                                             fpb       = 0,
                                                                   std::tuple<unsigned int, unsigned int, unsigned int> block_dim = {0, 0, 0},
                                                                   complex_layout                                       layout    = complex_layout::natural,
                                                                   real_mode                                            rmode     = real_mode::normal,
                                                                   experimental::code_type                              code_type = cufftdx::experimental::code_type::ptx) {

            bool has_block_dim = std::get<0>(block_dim) > 0 || std::get<1>(block_dim) > 0 || std::get<2>(block_dim) > 0;
            auto backend       = frontend_to_backend(algorithm::ct,
                                               execution,
                                               fft_size,
                                               type,
                                               dir,
                                               sm,
                                               rmode,
                                               fft_ept,
                                               std::get<0>(block_dim),
                                               code_type);

            auto query_result = database::detail::constexpr_db::runtime_query_database::query_runtime<checkLTODatabaseDefined>(
                backend.size, backend.type, backend.direction, backend.sm, prec, code_type, backend.elements_per_thread, backend.min_elements_per_thread);

            if (query_result.found) {
                const auto& selected_impl = *query_result.selected;
                return backend_to_frontend(execution,
                                           fft_ept > 0,
                                           type,
                                           layout,
                                           rmode,
                                           prec,
                                           fpb,
                                           backend.size,
                                           selected_impl.elements_per_thread,
                                           selected_impl.ffts_per_block,
                                           selected_impl.shared_memory_size,
                                           selected_impl.storage_size,
                                           selected_impl.elements_per_thread,
                                           selected_impl.ffts_per_block,
                                           has_block_dim,
                                           has_block_dim ? std::get<0>(block_dim) : 0,
                                           has_block_dim ? std::get<1>(block_dim) : 0,
                                           has_block_dim ? std::get<2>(block_dim) : 0);
            }
            return std::nullopt;
        }

        template<bool checkLTODatabaseDefined = true>
        std::vector<frontend_impl_traits> get_all_implementations_offline(unsigned                                             fft_size,
                                                                          fft_direction                                        dir,
                                                                          fft_type                                             type,
                                                                          unsigned                                             sm,
                                                                          execution_type                                       execution,
                                                                          precision                                            prec      = precision::f32,
                                                                          unsigned                                             fft_ept   = 0,
                                                                          unsigned                                             fpb       = 0,
                                                                          std::tuple<unsigned int, unsigned int, unsigned int> block_dim = {0, 0, 0},
                                                                          complex_layout                                       layout    = complex_layout::natural,
                                                                          real_mode                                            rmode     = real_mode::normal,
                                                                          experimental::code_type                              code_type = cufftdx::experimental::code_type::ptx) {
            bool has_block_dim = std::get<0>(block_dim) > 0 || std::get<1>(block_dim) > 0 || std::get<2>(block_dim) > 0;
            auto backend       = frontend_to_backend(algorithm::ct,
                                               execution,
                                               fft_size,
                                               type,
                                               dir,
                                               sm,
                                               rmode,
                                               fft_ept,
                                               std::get<0>(block_dim),
                                               code_type);

            auto possible_implementations = database::detail::constexpr_db::runtime_query_database::query_all_entries_runtime<checkLTODatabaseDefined>(
                backend.size, backend.type, backend.direction, backend.sm, prec, code_type, backend.elements_per_thread, backend.min_elements_per_thread);

            std::vector<frontend_impl_traits> returned_traits;
            for (const auto& impl : possible_implementations) {
                returned_traits.push_back(backend_to_frontend(execution,
                                                              fft_ept > 0,
                                                              type,
                                                              layout,
                                                              rmode,
                                                              prec,
                                                              fpb,
                                                              backend.size,
                                                              impl.elements_per_thread,
                                                              impl.ffts_per_block,
                                                              impl.shared_memory_size,
                                                              impl.storage_size,
                                                              impl.elements_per_thread,
                                                              impl.ffts_per_block,
                                                              has_block_dim,
                                                              has_block_dim ? std::get<0>(block_dim) : 0,
                                                              has_block_dim ? std::get<1>(block_dim) : 0,
                                                              has_block_dim ? std::get<2>(block_dim) : 0));
            }
            return returned_traits;
        }
#endif // CUFFTDX_ENABLE_RUNTIME_DATABASE

#ifdef CUFFTDX_ENABLE_CUFFT_DEPENDENCY
    } // namespace detail

    namespace utils {
        std::vector<detail::frontend_impl_traits> query_all_cufft_implementations(
            unsigned                                             fft_size,
            fft_direction                                        dir,
            fft_type                                             type,
            unsigned                                             sm,
            detail::execution_type                               execution,
            precision                                            prec,
            complex_layout                                       layout,
            real_mode                                            rmode,
            unsigned                                             fft_ept,
            unsigned                                             ffts_per_block,
            std::tuple<unsigned int, unsigned int, unsigned int> block_dim);
    } // namespace utils

    namespace detail {
        inline std::optional<frontend_impl_traits> query_single_cufft_implementation(
            unsigned                                             fft_size,
            fft_direction                                        dir,
            fft_type                                             type,
            unsigned                                             sm,
            execution_type                                       execution,
            precision                                            prec           = precision::f32,
            complex_layout                                       layout         = complex_layout::natural,
            real_mode                                            rmode          = real_mode::normal,
            unsigned                                             fft_ept        = 0,
            unsigned                                             ffts_per_block = 0,
            std::tuple<unsigned int, unsigned int, unsigned int> block_dim      = {0, 0, 0}) {

            auto all = utils::query_all_cufft_implementations(
                fft_size, dir, type, sm, execution, prec, layout, rmode, fft_ept, ffts_per_block, block_dim);
            if (all.empty()) {
                return std::nullopt;
            }
            return all[0];
        }
#endif // CUFFTDX_ENABLE_CUFFT_DEPENDENCY

    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_RUNTIME_QUERY_HELPERS_HPP
