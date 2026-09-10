# Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was mathdx-config.cmake.in                            ########

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

if(NOT TARGET mathdx::mathdx)
    # CUTLASS
    set_and_check(mathdx_included_CUTLASS_ROOT ${PACKAGE_PREFIX_DIR}/external/cutlass)
    set_and_check(mathdx_included_CUTLASS_INCLUDE_DIR ${PACKAGE_PREFIX_DIR}/external/cutlass/include)

    set(mathdx_DEPENDENCY_CUTLASS_RESOLVED FALSE)
    # Finds CUTLASS/CuTe and sets mathdx_cutlass_INCLUDE_DIR to <CUTLASS_ROOT>/include
    #
    # CUTLASS root directory is found by checking following variables in this order:
    # 1. find_package(NvidiaCutlass) (NvidiaCutlasss_ROOT)
    # 2. mathdx_CUTLASS_ROOT
    # 3. ENV{mathdx_CUTLASS_ROOT}
    # 4. find_package(NvidiaCutlass PATHS ${mathdx_included_CUTLASS_ROOT})
    set(mathdx_CUTLASS_MIN_VERSION 4.2.2)
    find_package(NvidiaCutlass QUIET)
    if(${NvidiaCutlass_FOUND})
        if(${NvidiaCutlass_VERSION} VERSION_LESS ${mathdx_CUTLASS_MIN_VERSION})
            message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${mathdx_CUTLASS_MIN_VERSION}")
        endif()
        get_property(mathdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        set_and_check(mathdx_cutlass_INCLUDE_DIR "${mathdx_NvidiaCutlass_include_dir}")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "mathdx: Found CUTLASS (NvidiaCutlass) dependency: ${mathdx_NvidiaCutlass_include_dir}")
        endif()
        set(mathdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED mathdx_CUTLASS_ROOT)
        get_filename_component(mathdx_CUTLASS_ROOT_ABSOLUTE ${mathdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(mathdx_cutlass_INCLUDE_DIR  "${mathdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "mathdx: Found CUTLASS dependency via mathdx_CUTLASS_ROOT: ${mathdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(mathdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    elseif(DEFINED ENV{mathdx_CUTLASS_ROOT})
        get_filename_component(mathdx_CUTLASS_ROOT_ABSOLUTE $ENV{mathdx_CUTLASS_ROOT} ABSOLUTE)
        set_and_check(mathdx_cutlass_INCLUDE_DIR "${mathdx_CUTLASS_ROOT_ABSOLUTE}/include")
        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "mathdx: Found CUTLASS dependency via ENV{mathdx_CUTLASS_ROOT}: ${mathdx_CUTLASS_ROOT_ABSOLUTE}")
        endif()
        set(mathdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
    else()
        find_package(NvidiaCutlass QUIET PATHS ${mathdx_included_CUTLASS_ROOT})
        if(${NvidiaCutlass_FOUND})
            if(${NvidiaCutlass_VERSION} VERSION_LESS ${mathdx_CUTLASS_MIN_VERSION})
                message(FATAL_ERROR "Found CUTLASS version is ${NvidiaCutlass_VERSION}, minimal required version is ${mathdx_CUTLASS_MIN_VERSION}")
            endif()
            get_property(mathdx_NvidiaCutlass_include_dir TARGET nvidia::cutlass::cutlass PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            set_and_check(mathdx_cutlass_INCLUDE_DIR "${mathdx_NvidiaCutlass_include_dir}")
            if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
                message(STATUS "mathdx: Found CUTLASS (NvidiaCutlass) dependency: ${mathdx_NvidiaCutlass_include_dir}")
            endif()
            set(mathdx_DEPENDENCY_CUTLASS_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${mathdx_DEPENDENCY_CUTLASS_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing CUTLASS dependency.\n"
                                "    You can set it via mathdx_CUTLASS_ROOT variable or by providing\n"
                                "    path to NvidiaCutlass package using NvidiaCutlass_ROOT or NvidiaCutlass_DIR.\n")
        endif()
    endif()

    # commondx
    find_package(commondx
        REQUIRED
        QUIET
        CONFIG
        PATHS "${PACKAGE_PREFIX_DIR}/lib/cmake/commondx/"
        NO_DEFAULT_PATH
    )

    # mathDx include directories
    set_and_check(mathdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
    set_and_check(mathdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
    # build: 9
    include("${CMAKE_CURRENT_LIST_DIR}/mathdx-targets.cmake")
endif()

list(TRANSFORM mathdx_FIND_COMPONENTS TOLOWER)
# Populate components list when blank or ALL is provided
set(_mathdx_all_components cufftdx cublasdx curanddx cusolverdx nvcompdx)
if(NOT mathdx_FIND_COMPONENTS OR "ALL" IN_LIST mathdx_FIND_COMPONENTS)
    if(NOT mathdx_FIND_COMPONENTS AND mathdx_FIND_REQUIRED)
        set(mathdx_FIND_REQUIRED_ALL TRUE)
    endif()
    set(mathdx_ALL_COMPONENTS TRUE)
    set(mathdx_FIND_COMPONENTS "")

    foreach(comp IN LISTS _mathdx_all_components)
        list(APPEND mathdx_FIND_COMPONENTS ${comp})
        if(mathdx_FIND_REQUIRED_ALL)
            set(mathdx_FIND_REQUIRED_${comp} TRUE)
        endif()
    endforeach()
endif()

# _mathdx_find_component(<comp> [<alias> <source>]...)        (private helper)
#
# Resolves a bundled mathDx sub-component via its own CMake config package.
# Sets mathdx::<comp> (alias of <comp>::<comp>) and mathdx_<comp>_FOUND.
#
# Optional ARGN: zero or more <alias> <source> pairs for additional imported-target
# aliases beyond the primary mathdx::<comp> one. Each pair creates:
#   add_library(<alias> ALIAS <source>)
# guarded by NOT TARGET <alias> AND TARGET <source>.
# ARGN must contain an even number of arguments (complete alias/source pairs).
macro(_mathdx_find_component comp)
    set(_mathdx_extras ${ARGN})
    list(LENGTH _mathdx_extras _mathdx_extras_len)
    math(EXPR _mathdx_extras_rem "${_mathdx_extras_len} % 2")
    # Validate upfront regardless of whether the component is active — bad call sites
    # should fail immediately, not silently pass when the component is not requested.
    if(_mathdx_extras_rem)
        message(FATAL_ERROR
            "_mathdx_find_component(${comp}): ARGN must contain <alias> <source> pairs "
            "(even count); got ${_mathdx_extras_len} argument(s).")
    endif()

    if("${comp}" IN_LIST mathdx_FIND_COMPONENTS)
        set(_mathdx_req "")
        if(mathdx_FIND_REQUIRED_${comp})
            set(_mathdx_req "REQUIRED")
        endif()
        set(_mathdx_quiet "")
        if(mathdx_FIND_QUIETLY)
            set(_mathdx_quiet "QUIET")
        endif()
        find_package(${comp}
            ${_mathdx_req}
            ${_mathdx_quiet}
            CONFIG
            PATHS "${PACKAGE_PREFIX_DIR}/lib/cmake/${comp}/"
            NO_DEFAULT_PATH
        )
        if(NOT TARGET mathdx::${comp})
            if(${comp}_FOUND)
                set(mathdx_${comp}_FOUND TRUE)
                add_library(mathdx::${comp} ALIAS ${comp}::${comp})
                while(_mathdx_extras)
                    list(POP_FRONT _mathdx_extras _mathdx_alias _mathdx_src)
                    if(NOT TARGET ${_mathdx_alias} AND TARGET ${_mathdx_src})
                        add_library(${_mathdx_alias} ALIAS ${_mathdx_src})
                    endif()
                endwhile()
                if(NOT mathdx_FIND_QUIETLY)
                    message(STATUS "mathDx: ${comp} found: ${${comp}_INCLUDE_DIRS}")
                endif()
            else()
                set(mathdx_${comp}_FOUND FALSE)
            endif()
        endif()
    endif()
    unset(_mathdx_extras)
    unset(_mathdx_extras_len)
    unset(_mathdx_extras_rem)
    unset(_mathdx_alias)
    unset(_mathdx_src)
    unset(_mathdx_req)
    unset(_mathdx_quiet)
endmacro()

# cuFFTDx
_mathdx_find_component(cufftdx
    # cufftdx_separate_twiddles_lut not cufftdx::cufftdx_separate_twiddles_lut
    # because it's an alias (and you can't alias an alias)
    mathdx::cufftdx_separate_twiddles_lut cufftdx_separate_twiddles_lut
)

# cuBLASDx
_mathdx_find_component(cublasdx
    mathdx::cublasdx_fatbin cublasdx::cublasdx_fatbin
    mathdx::cublasdx_no_lto cublasdx::cublasdx_no_lto
)

# cuRANDDx
_mathdx_find_component(curanddx)

# cuSOLVERDx
_mathdx_find_component(cusolverdx
    # cusolverdx_fatbin not cusolverdx::cusolverdx_fatbin
    # because it's an alias (and you can't alias an alias)
    mathdx::cusolverdx_fatbin cusolverdx_fatbin
)
if(cusolverdx_FOUND)
    set(cusolverdx_LIBRARIES mathdx::cusolverdx)
endif()

# nvCOMPDx
_mathdx_find_component(nvcompdx
    # nvcompdx_fatbin not mathdx::nvcompdx_fatbin
    # because it's an alias (and you can't alias an alias)
    mathdx::nvcompdx_fatbin nvcompdx_fatbin
)
if(nvcompdx_FOUND)
    set(nvcompdx_LIBRARIES mathdx::nvcompdx)
endif()

unset(_mathdx_all_components)
unset(mathdx_FIND_REQUIRED_ALL)

# Check all components
check_required_components(mathdx)
