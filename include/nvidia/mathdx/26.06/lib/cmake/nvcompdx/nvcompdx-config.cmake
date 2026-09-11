# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was nvcompdx-config.cmake.in                            ########

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

if(NOT TARGET nvcompdx::nvcompdx)
    # Find commondx
    set(nvcompdx_DEPENDENCY_COMMONDX_RESOLVED FALSE)
    if(TARGET commondx::commondx)
        set(nvcompdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
    elseif(NOT TARGET commondx::commondx)
        find_package(commondx QUIET)
        if(${commondx_FOUND})
            set(nvcompdx_DEPENDENCY_COMMONDX_RESOLVED TRUE)
        endif()
    endif()
    if(NOT ${nvcompdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
        if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "${CMAKE_FIND_PACKAGE_NAME} package NOT FOUND - dependency missing:\n"
                                "    Missing commonDx dependency.\n")
        endif()
    else()
        get_property(nvcompdx_commondx_include_dirs TARGET commondx::commondx PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(GET nvcompdx_commondx_include_dirs 0 nvcompdx_commondx_include_dir)
        set_and_check(nvcompdx_commondx_INCLUDE_DIR "${nvcompdx_commondx_include_dir}")

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "nvcompdx: Found commondx dependency")
        endif()
    endif()

    if(${nvcompdx_DEPENDENCY_COMMONDX_RESOLVED})
        set(nvcompdx_VERSION "0.1.4")
        # build: 9
        include("${CMAKE_CURRENT_LIST_DIR}/nvcompdx-targets.cmake")

        # Wrapper for the fatbin library
        if(NOT TARGET nvcompdx_fatbin)
            add_library(nvcompdx_fatbin INTERFACE)
            add_library(nvcompdx::nvcompdx_fatbin ALIAS nvcompdx_fatbin)
            target_compile_features(nvcompdx_fatbin INTERFACE cxx_std_17)
            target_compile_features(nvcompdx_fatbin INTERFACE cuda_std_17)
            set_and_check(nvcompdx_FATBIN "${PACKAGE_PREFIX_DIR}/lib/libnvcompdx.fatbin")
            if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
                target_link_libraries(nvcompdx_fatbin INTERFACE "-Xnvlink=-dlto,${nvcompdx_FATBIN}")
            else()
                target_link_options(nvcompdx_fatbin INTERFACE $<DEVICE_LINK:${nvcompdx_FATBIN}>)
            endif()
            target_include_directories(nvcompdx_fatbin INTERFACE "${PACKAGE_PREFIX_DIR}/include")
        endif()

        # Resolve dependencies:
        # 1) commondx
        target_link_libraries(nvcompdx::nvcompdx INTERFACE commondx::commondx)
        if(TARGET nvcompdx_fatbin)
            target_link_libraries(nvcompdx_fatbin INTERFACE commondx::commondx)
        endif()

        set_and_check(nvcompdx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(nvcompdx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
        set_and_check(nvcompdx_LIBRARY_DIRS "${PACKAGE_PREFIX_DIR}/lib")
        set(nvcompdx_LIBRARIES nvcompdx::nvcompdx)
        check_required_components(nvcompdx)

        if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
            message(STATUS "Found nvcompdx: (Version: 0.1.4, Include dirs: ${nvcompdx_INCLUDE_DIRS})")
        endif()
    else()
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    endif()
endif()
