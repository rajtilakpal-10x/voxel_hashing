cmake_minimum_required(VERSION 3.20)
project(voxel_hashing LANGUAGES CXX CUDA)


# -------------------------------
# Voxel Hashing Shared Lib
# -------------------------------
add_library(voxhash SHARED
    src/core/cuda_utils.cu
    src/core/vector.cu
    src/core/block.cu
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
