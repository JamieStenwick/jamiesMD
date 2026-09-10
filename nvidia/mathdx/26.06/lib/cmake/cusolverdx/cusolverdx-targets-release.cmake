#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cusolverdx::cusolverdx" for configuration "Release"
set_property(TARGET cusolverdx::cusolverdx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cusolverdx::cusolverdx PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcusolverdx.a"
  )

list(APPEND _cmake_import_check_targets cusolverdx::cusolverdx )
list(APPEND _cmake_import_check_files_for_cusolverdx::cusolverdx "${_IMPORT_PREFIX}/lib/libcusolverdx.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
