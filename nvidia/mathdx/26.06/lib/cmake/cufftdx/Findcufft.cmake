# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
if(NOT TARGET cufft::cufft)

    # Find CUDAToolkit
    set(cufft_DEPENDENCY_CUDATOOLKIT_RESOLVED FALSE)
    if(TARGET CUDA::cudart)
        set(cufft_DEPENDENCY_CUDATOOLKIT_RESOLVED TRUE)
    else()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(cufft_DEPENDENCY_CUDATOOLKIT_RESOLVED TRUE)
        endif()
    endif()
    if(NOT cufft_DEPENDENCY_CUDATOOLKIT_RESOLVED)
        message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                            "    Missing CUDAToolkit dependency.\n")
    else()
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cufft: Found CUDAToolkit dependency. Include dirs: ${CUDAToolkit_INCLUDE_DIRS}.")
        endif()
    endif()

    if(${cufft_DEPENDENCY_CUDATOOLKIT_RESOLVED})
        find_path(cufft_INCLUDE_DIR
            NAMES cufft.h
            PATHS
                ${cufft_ROOT}
                ${cufftdx_cufft_HOME}
                ${cufftdx_INCLUDE_DIRS}/../external/cufft
                ${CUDAToolkit_INCLUDE_DIRS}
            PATH_SUFFIXES
                include
            REQUIRED
        )

        # Extract version from cufft.h
        if(cufft_INCLUDE_DIR AND EXISTS "${cufft_INCLUDE_DIR}/cufft.h")
            file(STRINGS "${cufft_INCLUDE_DIR}/cufft.h" cufft_VERSION_DEFINE
                 REGEX "^#define[ \t]+CUFFT_VERSION[ \t]+[0-9]+")
            if(cufft_VERSION_DEFINE)
                string(REGEX REPLACE "^#define[ \t]+CUFFT_VERSION[ \t]+([0-9]+).*" "\\1"
                       cufft_VERSION_INT "${cufft_VERSION_DEFINE}")
                # CUFFT_VERSION is encoded as major*1000 + minor*100 + patch
                math(EXPR cufft_VERSION_MAJOR "${cufft_VERSION_INT} / 1000")
                math(EXPR cufft_VERSION_MINOR "(${cufft_VERSION_INT} % 1000) / 100")
                math(EXPR cufft_VERSION_PATCH "${cufft_VERSION_INT} % 100")
                set(cufft_VERSION "${cufft_VERSION_MAJOR}.${cufft_VERSION_MINOR}.${cufft_VERSION_PATCH}")
            endif()
        endif()

        # Extract Device API version from cufft_device.h (if available)
        if(cufft_INCLUDE_DIR AND EXISTS "${cufft_INCLUDE_DIR}/cufft_device.h")
            file(STRINGS "${cufft_INCLUDE_DIR}/cufft_device.h" cufft_DEVICE_VERSION_DEFINE
                 REGEX "^#define[ \t]+CUFFT_DEVICE_VERSION[ \t]+[0-9]+")
            if(cufft_DEVICE_VERSION_DEFINE)
                string(REGEX REPLACE "^#define[ \t]+CUFFT_DEVICE_VERSION[ \t]+([0-9]+).*" "\\1"
                       cufft_DEVICE_VERSION_INT "${cufft_DEVICE_VERSION_DEFINE}")
                # CUFFT_DEVICE_VERSION is encoded as major*1000 + minor*100 + patch
                math(EXPR cufft_DEVICE_VERSION_MAJOR "${cufft_DEVICE_VERSION_INT} / 1000")
                math(EXPR cufft_DEVICE_VERSION_MINOR "(${cufft_DEVICE_VERSION_INT} % 1000) / 100")
                math(EXPR cufft_DEVICE_VERSION_PATCH "${cufft_DEVICE_VERSION_INT} % 100")
                set(cufft_DEVICE_VERSION "${cufft_DEVICE_VERSION_MAJOR}.${cufft_DEVICE_VERSION_MINOR}.${cufft_DEVICE_VERSION_PATCH}")
            endif()
        endif()

        find_library(cufft_LIBRARY
            NAMES cufft
            PATHS "${cufft_INCLUDE_DIR}/../"
            PATH_SUFFIXES
                lib
                lib64
            NO_DEFAULT_PATH
            REQUIRED
        )
        find_library(cufft_static_LIBRARY
            NAMES cufft_static
            PATHS "${cufft_INCLUDE_DIR}/../"
            PATH_SUFFIXES
                lib
                lib64
            NO_DEFAULT_PATH
            REQUIRED
        )

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(cufft
            REQUIRED_VARS cufft_LIBRARY cufft_static_LIBRARY cufft_INCLUDE_DIR
            VERSION_VAR cufft_VERSION
        )
        if(cufft_FOUND)
            add_library(cufft::cufft SHARED IMPORTED)
            set_target_properties(cufft::cufft
                PROPERTIES
                    IMPORTED_LOCATION "${cufft_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${cufft_INCLUDE_DIR}"
            )
            target_include_directories(cufft::cufft
                INTERFACE
                    ${CUDAToolkit_INCLUDE_DIRS}
            )
            add_library(cufft::cufft_static STATIC IMPORTED)
            set_target_properties(cufft::cufft_static
                PROPERTIES
                    IMPORTED_LOCATION "${cufft_static_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${cufft_INCLUDE_DIR}"
            )
            target_include_directories(cufft::cufft_static
                INTERFACE
                    ${CUDAToolkit_INCLUDE_DIRS}
            )
            target_link_libraries(cufft::cufft_static
                INTERFACE
                    CUDA::culibos
                    CUDA::cudart_static_deps
            )
        endif()
        set(cufft_INCLUDE_DIRS "${cufft_INCLUDE_DIR}")
        set(cufft_LIBRARIES cufft::cufft cufft::cufft_static)
        mark_as_advanced(cufft_LIBRARY cufft_static_LIBRARY cufft_INCLUDE_DIR cufft_INCLUDE_DIRS cufft_LIBRARIES)
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            if(cufft_VERSION)
                set(_version_msg "Found cufft: ${cufft_VERSION}")
                if(cufft_DEVICE_VERSION)
                    string(APPEND _version_msg " (Device API: ${cufft_DEVICE_VERSION}")
                    string(APPEND _version_msg ")")
                endif()
                string(APPEND _version_msg " (Libraries: ${cufft_LIBRARIES}, Include dirs: ${cufft_INCLUDE_DIRS})")
                message(STATUS "${_version_msg}")
            else()
                message(STATUS "Found cufft: (Libraries: ${cufft_LIBRARIES}, Include dirs: ${cufft_INCLUDE_DIRS})")
            endif()
        endif()
    endif()
endif()
