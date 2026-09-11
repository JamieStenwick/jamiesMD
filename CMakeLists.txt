cmake_minimum_required(VERSION 3.24)

project(JamieSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(jamiesMD
    src/main.cpp
    src/Simulator.cpp
    src/CudaKernels.cu
)

find_package(mathdx REQUIRED COMPONENTS curanddx CONFIG PATHS "include/nvidia/mathdx/26.06/")
target_link_libraries(jamiesMD PRIVATE mathdx::curanddx)

set_target_properties(
    jamiesMD
    PROPERTIES
    CUDA_ARCHITECTURES 86
)

target_include_directories(jamiesMD PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
