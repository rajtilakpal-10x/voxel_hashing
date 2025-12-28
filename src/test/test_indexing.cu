#include <gtest/gtest.h>
#include <voxhash/core/cuda_utils.h>
#include <voxhash/core/indexing.h>
#include <voxhash/core/vector.h>

using namespace voxhash;

void EXPECT_INDEX_EQ(const Index3D& i1, const Index3D& i2) {
    EXPECT_EQ(i1.x, i2.x);
    EXPECT_EQ(i1.y, i2.y);
    EXPECT_EQ(i1.z, i2.z);
}

void EXPECT_VECTOR3F_EQ(const Vector3f& v1, const Vector3f& v2) {
    EXPECT_FLOAT_EQ(v1.x, v2.x);
    EXPECT_FLOAT_EQ(v1.y, v2.y);
    EXPECT_FLOAT_EQ(v1.z, v2.z);
}

TEST(TestIndexing, BlockAndVoxelSize) {
    const float block_size = 0.4f;
    EXPECT_FLOAT_EQ(blockSizeToVoxelSize(block_size), 0.05f);
    EXPECT_FLOAT_EQ(voxelSizeToBlockSize(blockSizeToVoxelSize(block_size)), block_size);
}

TEST(TestIndexing, BlockIdxFromPos) {
    const float block_size = 0.4f;
    Vector3f test_position(1.0f, 2.0f, 3.0f);
    Index3D bl_idx = getBlockIndexFromPosition(block_size, test_position);
    EXPECT_INDEX_EQ(bl_idx, Index3D(2, 5, 7));
}

TEST(TestIndexing, BlockPosFromIdx) {
    const float block_size = 0.4f;
    Index3D test_block_idx(2, 5, 7);
    Vector3f bl_pos = getPositionFromBlockIndex(block_size, test_block_idx);
    EXPECT_VECTOR3F_EQ(bl_pos, Vector3f(0.8f, 2.0f, 2.8f));
    Vector3f bl_center_pos = getCenterPositionFromBlockIndex(block_size, test_block_idx);
    EXPECT_VECTOR3F_EQ(bl_center_pos, Vector3f(1.0f, 2.2f, 3.0f));
}

TEST(TestIndexing, VoxelAndBlockIdxFromPos) {
    const float block_size = 0.4f;
    Vector3f test_position(1.0f, 2.0f, 3.0f);
    Index3D block_idx(0, 0, 0), voxel_idx(0, 0, 0);
    getBlockAndVoxelIndexFromPosition(block_size, test_position, &block_idx, &voxel_idx);
    EXPECT_INDEX_EQ(block_idx, Index3D(2, 5, 7));
    EXPECT_INDEX_EQ(voxel_idx, Index3D(3, 0, 4));
}

TEST(TestIndexing, VoxelAndBlockPosFromIdx) {
    const float block_size = 0.4f;
    Index3D block_idx(2, 5, 7), voxel_idx(3, 0, 4);
    Vector3f vbl_pos = getPositionFromBlockAndVoxelIndex(block_size, block_idx, voxel_idx);
    EXPECT_VECTOR3F_EQ(vbl_pos, Vector3f(0.95f, 2.0f, 3.0f));
    Vector3f vbl_center_pos =
            getCenterPositionFromBlockAndVoxelIndex(block_size, block_idx, voxel_idx);
    EXPECT_VECTOR3F_EQ(vbl_center_pos, Vector3f(0.975f, 2.025f, 3.025f));
}

TEST(TestIndexing, VoxelAndBlockIndicesFromPoses) {
    const float block_size = 0.4;
    std::vector<Vector3f> positions_to_test;
    positions_to_test.push_back(Vector3f(1.0f, 2.0f, 3.025f));
    positions_to_test.push_back(Vector3f(1.0f, 2.2f, 3.5f));
    positions_to_test.push_back(Vector3f(0.5f, 2.2f, 4.0f));

    CudaStreamOwning cuda_stream;
    Vector<Vector3f>::Ptr positions_vec =
            Vector<Vector3f>::copyFrom(positions_to_test, MemoryType::kDevice, cuda_stream);

    auto block_and_voxel_indices =
            getBlockAndVoxelIndicesFromPositions(block_size, *positions_vec, cuda_stream);

    Vector<Index3D>::Ptr block_indices = Vector<Index3D>::copyFrom(
            block_and_voxel_indices.first, MemoryType::kHost, cuda_stream);
    Vector<Index3D>::Ptr voxel_indices = Vector<Index3D>::copyFrom(
            block_and_voxel_indices.second, MemoryType::kHost, cuda_stream);

    cuda_stream.synchronize();

    EXPECT_INDEX_EQ(block_indices->data()[0], Index3D(2, 5, 7));
    EXPECT_INDEX_EQ(voxel_indices->data()[0], Index3D(3, 0, 4));
    EXPECT_INDEX_EQ(block_indices->data()[1], Index3D(2, 5, 8));
    EXPECT_INDEX_EQ(voxel_indices->data()[1], Index3D(3, 4, 5));
    EXPECT_INDEX_EQ(block_indices->data()[2], Index3D(1, 5, 10));
    EXPECT_INDEX_EQ(voxel_indices->data()[2], Index3D(1, 4, 0));
}