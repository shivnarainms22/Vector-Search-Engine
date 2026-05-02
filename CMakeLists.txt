cmake_minimum_required(VERSION 3.18)
project(vector_search CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Core static library
add_library(vsearch STATIC
    src/vector_index.cu
    src/kernel_naive.cu
    src/kernel_smem.cu
    src/kernel_coalesced.cu
    src/kernel_warp.cu
)
target_include_directories(vsearch PUBLIC include)
target_link_libraries(vsearch PUBLIC CUDA::cudart)
set_target_properties(vsearch PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Python extension module
pybind11_add_module(_vsearch python/bindings.cpp)
target_link_libraries(_vsearch PRIVATE vsearch CUDA::cudart)

# Tests
enable_testing()

add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE vsearch CUDA::cudart)
add_test(NAME correctness COMMAND test_correctness)

add_executable(test_perf tests/test_perf.cu)
target_link_libraries(test_perf PRIVATE vsearch)
add_test(NAME perf COMMAND test_perf)
