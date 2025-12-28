cmake_minimum_required(VERSION 3.20)

find_package(GTest REQUIRED)
# -------------------------------
# CUDA test helper function
# -------------------------------
function(add_cuda_gtest test_name test_file)
  add_executable(${test_name} ${test_file})

  target_link_libraries(${test_name}
    PRIVATE
      voxhash
      GTest::gtest
      GTest::gtest_main
      cuda
  )

  set_target_properties(${test_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  gtest_discover_tests(${test_name})
endfunction()

add_cuda_gtest(test_warmup test_warmup.cu)
add_cuda_gtest(test_vector test_vector.cu)
add_cuda_gtest(test_block test_block.cu)
add_cuda_gtest(test_indexing test_indexing.cu)
add_cuda_gtest(test_pool test_pool.cu)

# add_executable(test_tsdf_block_voxels test_tsdf_block_voxels.cu)
# target_link_libraries(test_tsdf_block_voxels PUBLIC voxhash)

# add_executable(test_semantic_block_voxels test_semantic_block_voxels.cu)
# target_link_libraries(test_semantic_block_voxels PUBLIC voxhash)

# add_executable(test_hash test_hash.cu)
# target_link_libraries(test_hash PUBLIC voxhash)

# add_executable(test_layer test_layer.cu)
# target_link_libraries(test_layer PUBLIC voxhash)

# add_executable(test_voxel_layer test_voxel_layer.cu)
# target_link_libraries(test_voxel_layer PUBLIC voxhash)
