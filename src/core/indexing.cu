#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/indexing.h"

namespace voxhash {

__global__ void getBlockAndVoxelIndexFromPositionKernel(
        const float block_size,
        size_t num_indices,
        const Vector3f* positions_L,
        Index3D* block_indices,
        Index3D* voxel_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices)
        getBlockAndVoxelIndexFromPosition(
                block_size, positions_L[idx], block_indices + idx, voxel_indices + idx);
}

std::pair<Vector<Index3D>, Vector<Index3D>> getBlockAndVoxelIndicesFromPositions(
        const float block_size,
        const Vector<Vector3f>& positions_L,
        const CudaStream& cuda_stream) {
    Vector<Index3D> block_indices(positions_L.size(), positions_L.location());
    Vector<Index3D> voxel_indices(positions_L.size(), positions_L.location());

    if (positions_L.location() == MemoryType::kHost) {
        for (size_t i = 0; i < positions_L.size(); i++)
            getBlockAndVoxelIndexFromPosition(
                    block_size,
                    positions_L.data()[i],
                    block_indices.data() + i,
                    voxel_indices.data() + i);
    } else {
        // Run retrieval kernel
        constexpr int kNumThreads = 512;
        const int kNumBlocks = voxel_indices.size() / kNumThreads + 1;
        getBlockAndVoxelIndexFromPositionKernel<<<kNumBlocks, kNumThreads, 0, cuda_stream>>>(
                block_size,
                positions_L.size(),
                positions_L.data(),
                block_indices.data(),
                voxel_indices.data());
        CUDA_CHECK(cudaGetLastError());
        cuda_stream.synchronize();
    }

    return std::make_pair(std::move(block_indices), std::move(voxel_indices));
}

}  // namespace voxhash