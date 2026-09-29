cmake_minimum_required(VERSION 3.24)

project(JamieSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(JAMIESMD_CUDA_ARCH 86)

set(CMAKE_CUDA_ARCHITECTURES ${JAMIESMD_CUDA_ARCH})

add_executable(jamiesMD
    src/main.cpp

    src/HostCode/Simulator.cpp
    src/HostCode/SimUtilities.cpp

    src/DeviceCode/CudaWrappers.cu
    src/DeviceCode/CudaObjects/DeviceCells.cu
    src/DeviceCode/CudaObjects/DeviceLookupTables.cu
    src/DeviceCode/CudaObjects/DeviceParticles.cu
    src/DeviceCode/CudaObjects/DeviceParams.cu
    src/DeviceCode/CudaObjects/FrameWrites.cu
)

target_compile_definitions(jamiesMD PRIVATE
    JAMIESMD_CURANDDX_SM=${JAMIESMD_CUDA_ARCH}0
)

find_package(mathdx REQUIRED COMPONENTS curanddx CONFIG PATHS "include/nvidia/mathdx/26.06/")
target_link_libraries(jamiesMD PRIVATE mathdx::curanddx)

target_include_directories(jamiesMD PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DeviceIncludes
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DeviceIncludes/CudaObjects
    ${CMAKE_CURRENT_SOURCE_DIR}/include/HostIncludes
)
