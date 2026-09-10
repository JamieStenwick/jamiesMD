# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cufftdx-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET cufftdx::cufftdx)
    # Find commondx
    set(cufftdx_DEPENDENCY_COMMONDX_RESOLVED FALSE)
    if(TARGET commondx::commondx)
        set(cufftdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(NOT TARGET commondx::commondx)
        find_package(commondx QUIET)
        if(${commondx_FOUND})
            set(cufftdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${cufftdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    else()
        get_property(cufftdx_commondx_include_dirs TARGET commondx::commondx PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(GET cufftdx_commondx_include_dirs 0 cufftdx_commondx_include_dir)
        set_and_check(cufftdx_commondx_INCLUDE_DIR "${cufftdx_commondx_include_dir}")

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cufftdx: Found commondx dependency")
        endif()
    endif()

    if(${cufftdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(cufftdx_VERSION "1.7.3")
        # build: 9
        include("${CMAKE_CURRENT_LIST_DIR}/cufftdx-targets.cmake")

        # Resolve dependencies:
        # 1) commondx
        target_link_libraries(cufftdx::cufftdx INTERFACE commondx::commondx)

        set_and_check(cufftdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(cufftdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(cufftdx_cufft_MODULE_PATH "${PACKAGE_PREFIX_DIR}/lib/cmake/cufftdx/")
        set(cufftdx_LIBRARIES cufftdx::cufftdx)
        check_required_components(cufftdx)

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "Found cufftdx: (Version: 1.7.3, Include dirs: ${cufftdx_INCLUDE_DIRS})")
        endif()
    else()
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    endif()
endif()

if(NOT TARGET cufftdx::cufftdx_separate_twiddles_lut)
    if(NOT DEFINED cufftdx_SEPARATE_TWIDDLES_CUDA_ARCHITECTURES OR cufftdx_SEPARATE_TWIDDLES_CUDA_ARCHITECTURES STREQUAL "")
        set(cufftdx_SEPARATE_TWIDDLES_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    endif()

    set_and_check(cufftdx_SEPARATE_TWIDDLES_SRCS "${cufftdx_INCLUDE_DIRS}/../src/cufftdx/lut.cu")
    add_library(cufftdx_separate_twiddles_lut OBJECT EXCLUDE_FROM_ALL ${cufftdx_SEPARATE_TWIDDLES_SRCS})
    add_library(cufftdx::cufftdx_separate_twiddles_lut ALIAS cufftdx_separate_twiddles_lut)
    set_target_properties(cufftdx_separate_twiddles_lut
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${cufftdx_SEPARATE_TWIDDLES_CUDA_ARCHITECTURES}"
    )
    target_compile_definitions(cufftdx_separate_twiddles_lut PUBLIC CUFFTDX_USE_SEPARATE_TWIDDLES)
    target_link_libraries(cufftdx_separate_twiddles_lut PUBLIC cufftdx::cufftdx)
endif()
