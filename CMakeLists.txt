cmake_minimum_required(VERSION 3.20)
project(voxel_hashing LANGUAGES CXX CUDA)

# -------------------------------
# Library: stdgpu
# -------------------------------

include(FetchContent)

FetchContent_Declare(
    stdgpu
    GIT_REPOSITORY https://github.com/stotko/stdgpu.git
    GIT_TAG        master
)

# Exclude unneeded parts from the build
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(STDGPU_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(stdgpu)

# Test stdgpu executable
add_executable(stdgpu_test src/test_unordered_map.cu)
target_link_libraries(stdgpu_test PUBLIC cuda cudart stdgpu::stdgpu)
set_target_properties(stdgpu_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# -------------------------------
# Voxel Hashing Shared Lib
# -------------------------------
add_library(voxhash SHARED
    src/core/cuda_utils.cu
    src/core/vector.cu
    src/core/block.cu
    src/core/memory_pool.cu
    src/core/layer.cu
    src/core/indexing.cu
)

target_link_libraries(voxhash PUBLIC cuda cudart)

target_include_directories(voxhash
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(voxhash PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_subdirectory(src/test)
