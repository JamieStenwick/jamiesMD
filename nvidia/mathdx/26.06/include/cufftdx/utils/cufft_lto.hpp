// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_UTILS_CUFFT_LTO_HPP
#define CUFFTDX_UTILS_CUFFT_LTO_HPP

#include <optional>
#include <vector>
#include <tuple>
#include <iostream>
#include <unordered_set>
#include <cufft_device.h>

// Check cuFFT Device API version
#define CUFFT_DEVICE_API_MIN_VERSION 200

#ifndef CUFFT_CHECK_AND_EXIT
#    define CUFFT_CHECK_AND_EXIT(error)                                                 \
        {                                                                               \
            auto status = static_cast<cufftResult>(error);                              \
            if (status != CUFFT_SUCCESS) {                                              \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUFFT_CHECK_AND_EXIT

namespace cufftdx {
    namespace utils {

        inline bool check_cufft_device_api_version() {
            int device_api_version = 0;
            CUFFT_CHECK_AND_EXIT(cufftDeviceGetVersion(&device_api_version));

            int major = device_api_version / 1000;
            int minor = (device_api_version % 1000) / 100;

            int major_min = CUFFT_DEVICE_API_MIN_VERSION / 1000;
            int minor_min = (CUFFT_DEVICE_API_MIN_VERSION % 1000) / 100;

            if (major != major_min || (major == major_min && minor < minor_min)) {
                std::cerr << "Error: cuFFT Device API minimum required version is not met or major version is not the same. Found version "
                          << major << "." << minor << ".*, but major version is " << major_min << " and minimum required version is " << major_min << "." << minor_min << ".*.\n";
                return false;
            }
            return true;
        }

        namespace cufft_detail {
            struct cufft_device_context {
                cufftDescriptionHandle desc_handle;
                cufftDeviceHandle      device_handle;
            };

            inline cufft_device_context create_cufft_device_context(
                const experimental::utils::backend_impl_traits& backend,
                precision                                      prec) {

                cufft_device_context ctx;
                CUFFT_CHECK_AND_EXIT(cufftDescriptionCreate(&ctx.desc_handle));
                CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(ctx.desc_handle, CUFFT_DESC_TRAIT_SIZE, static_cast<long long int>(backend.size)));
                CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(ctx.desc_handle, CUFFT_DESC_TRAIT_SM, static_cast<long long int>(backend.sm)));
                CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(ctx.desc_handle, CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD, static_cast<long long int>(backend.elements_per_thread)));

                auto cufft_dir = CUFFT_DESC_FORWARD;
                switch (backend.direction) {
                    case fft_direction::forward: cufft_dir = CUFFT_DESC_FORWARD; break;
                    case fft_direction::inverse: cufft_dir = CUFFT_DESC_INVERSE; break;
                }
                CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(ctx.desc_handle, CUFFT_DESC_TRAIT_DIRECTION, static_cast<long long int>(cufft_dir)));

                auto cufft_type = CUFFT_DESC_C2C;
                switch (backend.type) {
                    case fft_type::c2c: cufft_type = CUFFT_DESC_C2C; break;
                    case fft_type::c2r: cufft_type = CUFFT_DESC_C2R; break;
                    case fft_type::r2c: cufft_type = CUFFT_DESC_R2C; break;
                }
                CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(ctx.desc_handle, CUFFT_DESC_TRAIT_TYPE, static_cast<long long int>(cufft_type)));

                auto cufft_precision = CUFFT_DESC_SINGLE;
                switch (prec) {
                    case precision::f16: cufft_precision = CUFFT_DESC_HALF; break;
                    case precision::f32: cufft_precision = CUFFT_DESC_SINGLE; break;
                    case precision::f64: cufft_precision = CUFFT_DESC_DOUBLE; break;
                }
                CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(ctx.desc_handle, CUFFT_DESC_TRAIT_PRECISION, static_cast<long long int>(cufft_precision)));

                CUFFT_CHECK_AND_EXIT(cufftDeviceCreate(&ctx.device_handle, 1, &ctx.desc_handle));
                return ctx;
            }

        } // namespace cufft_detail

        inline std::tuple<std::string, std::vector<std::vector<char>>, dim3, unsigned int> get_database_and_ltoir(unsigned       fft_size,
                                                                                                                  fft_direction  dir,
                                                                                                                  fft_type       type,
                                                                                                                  unsigned       sm,
                                                                                                                  execution_type execution,
                                                                                                                  precision      prec           = precision::f32,
                                                                                                                  complex_layout layout         = complex_layout::natural,
                                                                                                                  real_mode      rmode          = real_mode::normal,
                                                                                                                  unsigned       fft_ept        = 0 /* use heuristic */,
                                                                                                                  unsigned       ffts_per_block = 1 /* 0: use suggested ffts_per_block */,
                                                                                                                  dim3           block_dim      = dim3(0, 0, 0)) {

            bool has_block_dim = block_dim.x > 0 || block_dim.y > 0 || block_dim.z > 0;

            std::vector<char>              database_str;
            std::vector<std::vector<char>> codes;

            dim3     result_block_dim(1, 1, 1);
            unsigned shared_memory_size = 0;

            auto backend = experimental::utils::frontend_to_backend(algorithm::ct,
                                                                    execution,
                                                                    fft_size,
                                                                    type,
                                                                    dir,
                                                                    sm,
                                                                    rmode,
                                                                    fft_ept,
                                                                    block_dim.x,
                                                                    experimental::code_type::ltoir);
            auto ctx     = cufft_detail::create_cufft_device_context(backend, prec);

            size_t func_count;
            CUFFT_CHECK_AND_EXIT(cufftDeviceGetNumDeviceFunctions(ctx.device_handle, ctx.desc_handle, &func_count));
            if (func_count > 0) {
                std::vector<cufftDeviceFunctionHandle> funcs(func_count);
                CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctions(ctx.device_handle, ctx.desc_handle, funcs.size(), funcs.data()));

                long long int func_elements_per_thread;
                CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctionTraitInt64(ctx.device_handle, funcs[0], CUFFT_DEVICE_FUNC_TRAIT_ELEMENTS_PER_THREAD, &func_elements_per_thread));

                auto ept       = static_cast<unsigned int>(func_elements_per_thread);
                bool ept_valid = (fft_ept > 0) ? (ept == backend.elements_per_thread)
                                               : (ept >= backend.min_elements_per_thread);

                if (ept_valid) {
                    long long int func_suggested_ffts_per_block;
                    long long int func_smem_per_fft;
                    long long int func_storage_size;
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctionTraitInt64(ctx.device_handle, funcs[0], CUFFT_DEVICE_FUNC_TRAIT_SUGGESTED_FFTS_PER_BLOCK, &func_suggested_ffts_per_block));
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctionTraitInt64(ctx.device_handle, funcs[0], CUFFT_DEVICE_FUNC_TRAIT_SHARED_MEMORY_PER_FFT, &func_smem_per_fft));
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctionTraitInt64(ctx.device_handle, funcs[0], CUFFT_DEVICE_FUNC_TRAIT_STORAGE_SIZE, &func_storage_size));

                    auto frontend = detail::backend_to_frontend(execution,
                                                                (fft_ept != 0),
                                                                type,
                                                                layout,
                                                                rmode,
                                                                prec,
                                                                ffts_per_block,
                                                                backend.size,
                                                                func_elements_per_thread,
                                                                func_suggested_ffts_per_block,
                                                                func_smem_per_fft,
                                                                func_storage_size,
                                                                func_elements_per_thread,
                                                                func_suggested_ffts_per_block,
                                                                has_block_dim,
                                                                has_block_dim ? block_dim.x : 0,
                                                                has_block_dim ? block_dim.y : 0,
                                                                has_block_dim ? block_dim.z : 0);

                    result_block_dim   = dim3(frontend.block_dim_x, frontend.block_dim_y, frontend.block_dim_z);
                    shared_memory_size = frontend.shared_memory_size;

                    size_t database_str_size = 0;
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDatabaseStrSize(ctx.device_handle, &database_str_size, funcs[0]));
                    if (database_str_size > 0) {
                        database_str.resize(database_str_size);
                        CUFFT_CHECK_AND_EXIT(cufftDeviceGetDatabaseStr(ctx.device_handle, database_str.size(), database_str.data(), funcs[0]));
                    }

                    size_t count = 0;
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetNumLTOIRs(ctx.device_handle, &count, funcs[0]));
                    if (count > 0) {
                        std::vector<size_t> code_sizes(count);
                        CUFFT_CHECK_AND_EXIT(cufftDeviceGetLTOIRSizes(ctx.device_handle, code_sizes.size(), code_sizes.data(), funcs[0]));
                        for (auto size : code_sizes) {
                            codes.push_back(std::vector<char>(size));
                        }
                        std::vector<char*> code_ptrs;
                        for (auto& code : codes) {
                            code_ptrs.push_back(code.data());
                        }
                        std::vector<cufftDeviceCodeContainer> code_containers(count);
                        CUFFT_CHECK_AND_EXIT(cufftDeviceGetLTOIRs(ctx.device_handle, code_ptrs.size(), code_ptrs.data(), code_containers.data(), funcs[0]));
                    }
                }
            }
            CUFFT_CHECK_AND_EXIT(cufftDeviceDestroy(ctx.device_handle));
            return std::make_tuple(database_str.size() > 0 ? std::string(database_str.data()) : std::string(),
                                   codes,
                                   result_block_dim,
                                   shared_memory_size);
        }

        inline std::vector<experimental::utils::frontend_impl_traits> query_all_cufft_implementations(
            unsigned                                             fft_size,
            fft_direction                                        dir,
            fft_type                                             type,
            unsigned                                             sm,
            execution_type                                       execution,
            precision                                            prec           = precision::f32,
            complex_layout                                       layout         = complex_layout::natural,
            real_mode                                            rmode          = real_mode::normal,
            unsigned                                             fft_ept        = 0 /* use heuristic */,
            unsigned                                             ffts_per_block = 1 /* 0: use suggested ffts_per_block */,
            std::tuple<unsigned int, unsigned int, unsigned int> block_dim      = {0, 0, 0}) {

            bool has_block_dim = std::get<0>(block_dim) > 0 || std::get<1>(block_dim) > 0 || std::get<2>(block_dim) > 0;

            auto backend = experimental::utils::frontend_to_backend(algorithm::ct,
                                                                    execution,
                                                                    fft_size,
                                                                    type,
                                                                    dir,
                                                                    sm,
                                                                    rmode,
                                                                    fft_ept,
                                                                    std::get<0>(block_dim),
                                                                    experimental::code_type::ltoir);
            auto ctx     = cufft_detail::create_cufft_device_context(backend, prec);

            std::vector<experimental::utils::frontend_impl_traits> results;
            std::unordered_set<unsigned int>                       seen_epts;

            size_t func_count;
            CUFFT_CHECK_AND_EXIT(cufftDeviceGetNumDeviceFunctions(ctx.device_handle, ctx.desc_handle, &func_count));
            if (func_count > 0) {
                std::vector<cufftDeviceFunctionHandle> funcs(func_count);
                CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctions(ctx.device_handle, ctx.desc_handle, funcs.size(), funcs.data()));

                for (size_t fi = 0; fi < func_count; ++fi) {
                    long long int func_elements_per_thread;
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctionTraitInt64(ctx.device_handle, funcs[fi], CUFFT_DEVICE_FUNC_TRAIT_ELEMENTS_PER_THREAD, &func_elements_per_thread));

                    auto ept = static_cast<unsigned int>(func_elements_per_thread);
                    if (seen_epts.count(ept))
                        continue;
                    if (fft_ept > 0 && ept != backend.elements_per_thread)
                        continue;
                    if (fft_ept == 0 && ept < backend.min_elements_per_thread)
                        continue;

                    long long int func_suggested_ffts_per_block;
                    long long int func_smem_per_fft;
                    long long int func_storage_size;
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctionTraitInt64(ctx.device_handle, funcs[fi], CUFFT_DEVICE_FUNC_TRAIT_SUGGESTED_FFTS_PER_BLOCK, &func_suggested_ffts_per_block));
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctionTraitInt64(ctx.device_handle, funcs[fi], CUFFT_DEVICE_FUNC_TRAIT_SHARED_MEMORY_PER_FFT, &func_smem_per_fft));
                    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDeviceFunctionTraitInt64(ctx.device_handle, funcs[fi], CUFFT_DEVICE_FUNC_TRAIT_STORAGE_SIZE, &func_storage_size));

                    seen_epts.insert(ept);
                    results.push_back(detail::backend_to_frontend(execution,
                                                                  (fft_ept != 0),
                                                                  type,
                                                                  layout,
                                                                  rmode,
                                                                  prec,
                                                                  ffts_per_block,
                                                                  backend.size,
                                                                  func_elements_per_thread,
                                                                  func_suggested_ffts_per_block,
                                                                  func_smem_per_fft,
                                                                  func_storage_size,
                                                                  func_elements_per_thread,
                                                                  func_suggested_ffts_per_block,
                                                                  has_block_dim,
                                                                  has_block_dim ? std::get<0>(block_dim) : 0,
                                                                  has_block_dim ? std::get<1>(block_dim) : 0,
                                                                  has_block_dim ? std::get<2>(block_dim) : 0));
                }
            }

            CUFFT_CHECK_AND_EXIT(cufftDeviceDestroy(ctx.device_handle));
            return results;
        }
    } // namespace utils
} // namespace cufftdx

#endif // CUFFTDX_UTILS_CUFFT_LTO_HPP
