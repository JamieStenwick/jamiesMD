# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cusolverdx-config.cmake.in                            ########

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

if(NOT TARGET cusolverdx::cusolverdx)
    # Find CUTLASS
    set(cusolverdx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    set(cusolverdx_CUTLASS_MIN_VERSION 4.4.1)
    find_package(NvidiaCutlass QUIET)
    if(${NvidiaCutlass_FOUND})
        if(${NvidiaCutlass_VERSION} VERSION_LESS ${cusolverdx_CUTLASS_MIN_VERSION})
            message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${cusolverdx_CUTLASS_MIN_VERSION}")
        endif()
        get_property(cusolverdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        set_and_check(cusolverdx_cutlass_INCLUDE_DIR "${cusolverdx_NvidiaCutlass_include_dir}")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cusolverdx: Found CUTLASS (NvidiaCutlass) dependency: ${cusolverdx_NvidiaCutlass_include_dir}")
        endif()
        set(cusolverdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED cusolverdx_CUTLASS_ROOT)
        get_filename_component(cusolverdx_CUTLASS_ROOT_ABSOLUTE ${cusolverdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(cusolverdx_cutlass_INCLUDE_DIR  "${cusolverdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cusolverdx: Found CUTLASS dependency via cusolverdx_CUTLASS_ROOT: ${cusolverdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(cusolverdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED ENV{cusolverdx_CUTLASS_ROOT})
        get_filename_component(cusolverdx_CUTLASS_ROOT_ABSOLUTE $ENV{cusolverdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(cusolverdx_cutlass_INCLUDE_DIR "${cusolverdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cusolverdx: Found CUTLASS dependency via ENV{cusolverdx_CUTLASS_ROOT}: ${cusolverdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(cusolverdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    endif()
    if(NOT ${cusolverdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing CUTLASS dependency.\n"
                                "    You can set it via cusolverdx_CUTLASS_ROOT variable or by providing\n"
                                "    path to NvidiaCutlass package using NvidiaCutlass_ROOT or NvidiaCutlass_DIR.\n")
        endif()
    endif()

    # Find commondx
    set(cusolverdx_DEPENDENCY_COMMONDX_RESOLVED FALSE)
    if(TARGET commondx::commondx)
        set(cusolverdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(NOT TARGET commondx::commondx)
        find_package(commondx QUIET)
        if(${commondx_FOUND})
            set(cusolverdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${cusolverdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    else()

        get_property(cusolverdx_commondx_include_dirs TARGET commondx::commondx PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(GET cusolverdx_commondx_include_dirs 0 cusolverdx_commondx_include_dir)
        set_and_check(cusolverdx_commondx_INCLUDE_DIR "${cusolverdx_commondx_include_dir}")

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cusolverdx: Found commondx dependency")
        endif()
    endif()

    if(${cusolverdx_DEPENDENCY_COMMONDX_RESOLVED} AND ${cusolverdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(cusolverdx_VERSION "0.5.0")
        # build: 9
        include("${CMAKE_CURRENT_LIST_DIR}/cusolverdx-targets.cmake")

        # Wrapper for fatbin library
        if(NOT TARGET cusolverdx_fatbin)
            add_library(cusolverdx_fatbin INTERFACE)
            add_library(cusolverdx::cusolverdx_fatbin ALIAS cusolverdx_fatbin)
            target_compile_features(cusolverdx_fatbin INTERFACE cxx_std_17)
            target_compile_features(cusolverdx_fatbin INTERFACE cuda_std_17)
            set_and_check(cusolverdx_FATBIN "${PACKAGE_PREFIX_DIR}/lib/libcusolverdx.fatbin")
            if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
                target_link_libraries(cusolverdx_fatbin INTERFACE "-Xnvlink=-dlto,${cusolverdx_FATBIN}")
            else()
                target_link_options(cusolverdx_fatbin INTERFACE $<DEVICE_LINK:${cusolverdx_FATBIN}>)
            endif()
            target_include_directories(cusolverdx_fatbin INTERFACE "${PACKAGE_PREFIX_DIR}/include")
        endif()

        # Resolve dependencies:
        # 1) CUTLASS
        if(${NvidiaCutlass_FOUND})
            target_link_libraries(cusolverdx::cusolverdx INTERFACE nvidia::cutlass::cutlass)
            if(TARGET cusolverdx_fatbin)
                target_link_libraries(cusolverdx_fatbin INTERFACE nvidia::cutlass::cutlass)
            endif()
        elseif(DEFINED cusolverdx_CUTLASS_ROOT)
            target_include_directories(cusolverdx::cusolverdx INTERFACE ${cusolverdx_cutlass_INCLUDE_DIR})
            if(TARGET cusolverdx_fatbin)
                target_include_directories(cusolverdx_fatbin INTERFACE ${cusolverdx_cutlass_INCLUDE_DIR})
            endif()
        elseif(DEFINED ENV{cusolverdx_CUTLASS_ROOT})
            target_include_directories(cusolverdx::cusolverdx INTERFACE ${cusolverdx_cutlass_INCLUDE_DIR})
            if(TARGET cusolverdx_fatbin)
                target_include_directories(cusolverdx_fatbin INTERFACE ${cusolverdx_cutlass_INCLUDE_DIR})
            endif()
        endif()
        # 2) commondx
        target_link_libraries(cusolverdx::cusolverdx INTERFACE commondx::commondx)
        if(TARGET cusolverdx_fatbin)
            target_link_libraries(cusolverdx_fatbin INTERFACE commondx::commondx)
        endif()

        set_and_check(cusolverdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(cusolverdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(cusolverdx_LIBRARY_DIRS "${PACKAGE_PREFIX_DIR}/lib")
        set(cusolverdx_LIBRARIES cusolverdx::cusolverdx)
        check_required_components(cusolverdx)

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "Found cusolverdx: (Version: 0.5.0, Include dirs: ${cusolverdx_INCLUDE_DIRS})")
        endif()
    else()
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    endif()
endif()
