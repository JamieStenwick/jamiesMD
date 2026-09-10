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

target_include_directories(jamiesMD PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
