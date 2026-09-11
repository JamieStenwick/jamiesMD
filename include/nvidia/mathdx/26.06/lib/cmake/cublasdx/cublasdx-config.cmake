# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cublasdx-config.cmake.in                            ########

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

if(NOT TARGET cublasdx::cublasdx)
    # Finds CUTLASS/CuTe and sets cublasdx_cutlass_INCLUDE_DIR to <CUTLASS_ROOT>/include
    #
    # CUTLASS root directory is found by checking following variables in this order:
    # 1. Root directory of NvidiaCutlass package
    # 2. cublasdx_CUTLASS_ROOT
    # 3. ENV{cublasdx_CUTLASS_ROOT}
    # 4. cublasdx_CUTLASS_ROOT
    set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    set(cublasdx_CUTLASS_MIN_VERSION 4.4.1)
    find_package(NvidiaCutlass QUIET)
    if(${NvidiaCutlass_FOUND})
        if(${NvidiaCutlass_VERSION} VERSION_LESS ${cublasdx_CUTLASS_MIN_VERSION})
            message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${cublasdx_CUTLASS_MIN_VERSION}")
        endif()
        get_property(cublasdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        set_and_check(cublasdx_cutlass_INCLUDE_DIR "${cublasdx_NvidiaCutlass_include_dir}")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cublasdx: Found CUTLASS (NvidiaCutlass) dependency: ${cublasdx_NvidiaCutlass_include_dir}")
        endif()
        set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED cublasdx_CUTLASS_ROOT)
        get_filename_component(cublasdx_CUTLASS_ROOT_ABSOLUTE ${cublasdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(cublasdx_cutlass_INCLUDE_DIR  "${cublasdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cublasdx: Found CUTLASS dependency via cublasdx_CUTLASS_ROOT: ${cublasdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED ENV{cublasdx_CUTLASS_ROOT})
        get_filename_component(cublasdx_CUTLASS_ROOT_ABSOLUTE $ENV{cublasdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(cublasdx_cutlass_INCLUDE_DIR "${cublasdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cublasdx: Found CUTLASS dependency via ENV{cublasdx_CUTLASS_ROOT}: ${cublasdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(cublasdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    endif()
    if(NOT ${cublasdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing CUTLASS dependency.\n"
                                "    You can set it via cublasdx_CUTLASS_ROOT variable or by providing\n"
                                "    path to NvidiaCutlass package using NvidiaCutlass_ROOT or NvidiaCutlass_DIR.\n")
        endif()
    endif()

    # Find commondx
    set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED FALSE)
    if(TARGET commondx::commondx)
        set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(NOT TARGET commondx::commondx)
        find_package(commondx QUIET)
        if(${commondx_FOUND})
            set(cublasdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${cublasdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    else()
        get_property(cublasdx_commondx_include_dirs TARGET commondx::commondx PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(GET cublasdx_commondx_include_dirs 0 cublasdx_commondx_include_dir)
        set_and_check(cublasdx_commondx_INCLUDE_DIR "${cublasdx_commondx_include_dir}")

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "cublasdx: Found commondx dependency")
        endif()
    endif()

    if(${cublasdx_DEPENDENCY_COMMONDX_RESOLVED} AND ${cublasdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(cublasdx_VERSION "0.7.1")
        # build: 9
        include("${CMAKE_CURRENT_LIST_DIR}/cublasdx-targets.cmake")

        set(cublasdx_CUTLASS_VERSION "${NvidiaCutlass_VERSION}")
        if(NOT cublasdx_CUTLASS_VERSION)
            file(READ "${cublasdx_cutlass_INCLUDE_DIR}/cutlass/version.h" cublasdx_CUTLASS_VERSION_H)
            string(REGEX MATCH "#define CUTLASS_MAJOR ([0-9]+)" _ "${cublasdx_CUTLASS_VERSION_H}")
            set(cublasdx_CUTLASS_VERSION_MAJOR "${CMAKE_MATCH_1}")
            string(REGEX MATCH "#define CUTLASS_MINOR ([0-9]+)" _ "${cublasdx_CUTLASS_VERSION_H}")
            set(cublasdx_CUTLASS_VERSION_MINOR "${CMAKE_MATCH_1}")
            string(REGEX MATCH "#define CUTLASS_PATCH ([0-9]+)" _ "${cublasdx_CUTLASS_VERSION_H}")
            set(cublasdx_CUTLASS_VERSION_PATCH "${CMAKE_MATCH_1}")
        elseif(cublasdx_CUTLASS_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
            set(cublasdx_CUTLASS_VERSION_MAJOR "${CMAKE_MATCH_1}")
            set(cublasdx_CUTLASS_VERSION_MINOR "${CMAKE_MATCH_2}")
            set(cublasdx_CUTLASS_VERSION_PATCH "${CMAKE_MATCH_3}")
        endif()
        if("${cublasdx_CUTLASS_VERSION_MAJOR}" STREQUAL "" OR
           "${cublasdx_CUTLASS_VERSION_MINOR}" STREQUAL "" OR
           "${cublasdx_CUTLASS_VERSION_PATCH}" STREQUAL "")
            message(FATAL_ERROR "Could not determine CUTLASS version")
        endif()
        math(EXPR cublasdx_CUTLASS_VERSION_NUMBER
             "${cublasdx_CUTLASS_VERSION_MAJOR} * 10000 + ${cublasdx_CUTLASS_VERSION_MINOR} * 100 + ${cublasdx_CUTLASS_VERSION_PATCH}")
        foreach(cublasdx_TARGET cublasdx::cublasdx cublasdx::cublasdx_fatbin cublasdx::cublasdx_no_lto)
            target_compile_definitions(${cublasdx_TARGET} INTERFACE CUBLASDX_CUTLASS_VERSION=${cublasdx_CUTLASS_VERSION_NUMBER})
        endforeach()

        set_and_check(cublasdx_FATBIN "${PACKAGE_PREFIX_DIR}/lib/libcublasdx.fatbin")

        # cublasdx lto options
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.2)
            target_link_options(cublasdx::cublasdx INTERFACE $<DEVICE_LINK:${cublasdx_FATBIN}>)
        else()
            target_compile_definitions(cublasdx::cublasdx INTERFACE CUBLASDX_NO_FATBIN_AVAILABLE)
        endif()

        # cublasdx_fatbin lto options
        target_link_options(cublasdx::cublasdx_fatbin INTERFACE $<DEVICE_LINK:${cublasdx_FATBIN}>)
        
        # cublasdx_no_lto lto options
        target_compile_definitions(cublasdx::cublasdx_no_lto INTERFACE CUBLASDX_NO_FATBIN_AVAILABLE)

        # Resolve dependencies:
        # 1) CUTLASS
        if(${NvidiaCutlass_FOUND})
            target_link_libraries(cublasdx::cublasdx INTERFACE nvidia::cutlass::cutlass)
            target_link_libraries(cublasdx::cublasdx_fatbin INTERFACE nvidia::cutlass::cutlass)
            target_link_libraries(cublasdx::cublasdx_no_lto INTERFACE nvidia::cutlass::cutlass)
        elseif(DEFINED cublasdx_CUTLASS_ROOT)
            target_include_directories(cublasdx::cublasdx INTERFACE ${cublasdx_cutlass_INCLUDE_DIR})
            target_include_directories(cublasdx::cublasdx_fatbin INTERFACE ${cublasdx_cutlass_INCLUDE_DIR})
            target_include_directories(cublasdx::cublasdx_no_lto INTERFACE ${cublasdx_cutlass_INCLUDE_DIR})
        elseif(DEFINED ENV{cublasdx_CUTLASS_ROOT})
            target_include_directories(cublasdx::cublasdx INTERFACE ${cublasdx_cutlass_INCLUDE_DIR})
            target_include_directories(cublasdx::cublasdx_fatbin INTERFACE ${cublasdx_cutlass_INCLUDE_DIR})
            target_include_directories(cublasdx::cublasdx_no_lto INTERFACE ${cublasdx_cutlass_INCLUDE_DIR})
        endif()
        # 2) commondx
        target_link_libraries(cublasdx::cublasdx INTERFACE commondx::commondx)
        target_link_libraries(cublasdx::cublasdx_fatbin INTERFACE commondx::commondx)
        target_link_libraries(cublasdx::cublasdx_no_lto INTERFACE commondx::commondx)

        set_and_check(cublasdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(cublasdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
        set(cublasdx_LIBRARIES cublasdx::cublasdx cublasdx::cublasdx_fatbin cublasdx::cublasdx_no_lto)
        check_required_components(cublasdx)

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "Found cublasdx: (Version: 0.7.1, Include dirs: ${cublasdx_INCLUDE_DIRS})")
        endif()
    else()
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    endif()
endif()
