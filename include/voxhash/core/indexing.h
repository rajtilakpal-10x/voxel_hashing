#pragma once

#include "voxhash/core/block.h"

namespace voxhash {

template <typename T>
__host__ __device__ inline T hd_min(T a, T b) {
    return (a < b) ? a : b;
}

__host__ __device__ inline float voxelSizeToBlockSize(const float voxel_size) {
    return voxel_size * Block<bool>::kVoxelsPerSide;
}

__host__ __device__ inline float blockSizeToVoxelSize(const float block_size) {
    return block_size / Block<bool>::kVoxelsPerSide;
}

__host__ __device__ inline Index3D getBlockIndexFromPosition(
        const float block_size, const Vector3f& position_L) {
    int x = (int)(position_L.x / block_size);
    int y = (int)(position_L.y / block_size);
    int z = (int)(position_L.z / block_size);

    return Index3D(x, y, z);
}

__host__ __device__ inline Vector3f getPositionFromBlockIndex(
        const float block_size, const Index3D& index) {
    float x = index.x * block_size;
    float y = index.y * block_size;
    float z = index.z * block_size;
    return Vector3f(x, y, z);
}

__host__ __device__ inline Vector3f getCenterPositionFromBlockIndex(
        const float block_size, const Index3D& index) {
    float x = (index.x + 0.5f) * block_size;
    float y = (index.y + 0.5f) * block_size;
    float z = (index.z + 0.5f) * block_size;
    return Vector3f(x, y, z);
}

__host__ __device__ inline void getBlockAndVoxelIndexFromPosition(
        const float block_size,
        const Vector3f& position_L,
        Index3D* block_idx,
        Index3D* voxel_idx) {
    *block_idx = getBlockIndexFromPosition(block_size, position_L);
    const float voxel_size = blockSizeToVoxelSize(block_size);
    const int kVoxelsPerSideMinusOne = Block<bool>::kVoxelsPerSide - 1;

    int v_x = hd_min(
            kVoxelsPerSideMinusOne, (int)((position_L.x - block_idx->x * block_size) / voxel_size));

    int v_y = hd_min(
            kVoxelsPerSideMinusOne, (int)((position_L.y - block_idx->y * block_size) / voxel_size));

    int v_z = hd_min(
            kVoxelsPerSideMinusOne, (int)((position_L.z - block_idx->z * block_size) / voxel_size));
    *voxel_idx = Index3D(v_x, v_y, v_z);
}

__host__ __device__ inline Vector3f getPositionFromBlockAndVoxelIndex(
        const float block_size, const Index3D& block_idx, const Index3D& voxel_idx) {
    const float voxel_size = blockSizeToVoxelSize(block_size);
    float p_x = block_idx.x * block_size + voxel_idx.x * voxel_size;
    float p_y = block_idx.y * block_size + voxel_idx.y * voxel_size;
    float p_z = block_idx.z * block_size + voxel_idx.z * voxel_size;
    return Vector3f(p_x, p_y, p_z);
}

__host__ __device__ inline Vector3f getCenterPositionFromBlockAndVoxelIndex(
        const float block_size, const Index3D& block_idx, const Index3D& voxel_idx) {
    const float voxel_size = blockSizeToVoxelSize(block_size);
    float p_x = block_idx.x * block_size + voxel_idx.x * voxel_size + 0.5f * voxel_size;
    float p_y = block_idx.y * block_size + voxel_idx.y * voxel_size + 0.5f * voxel_size;
    float p_z = block_idx.z * block_size + voxel_idx.z * voxel_size + 0.5f * voxel_size;
    return Vector3f(p_x, p_y, p_z);
}

std::pair<Vector<Index3D>, Vector<Index3D>> getBlockAndVoxelIndicesFromPositions(
        const float block_size,
        const Vector<Vector3f>& positions_L,
        const CudaStream& cuda_stream = CudaStreamOwning());
// TODO: Add other bulk methods as necessary
}  // namespace voxhash