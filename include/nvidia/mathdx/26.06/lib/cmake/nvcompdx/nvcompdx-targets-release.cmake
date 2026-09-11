#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nvcompdx::nvcompdx" for configuration "Release"
set_property(TARGET nvcompdx::nvcompdx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nvcompdx::nvcompdx PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libnvcompdx.a"
  )

list(APPEND _cmake_import_check_targets nvcompdx::nvcompdx )
list(APPEND _cmake_import_check_files_for_nvcompdx::nvcompdx "${_IMPORT_PREFIX}/lib/libnvcompdx.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
