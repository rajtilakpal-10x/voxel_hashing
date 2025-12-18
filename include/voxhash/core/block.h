#pragma once

#include "voxhash/core/vector.h"

namespace voxhash
{

    struct Index3D
    {
        size_t x, y, z;
        Index3D(size_t x, size_t y, size_t z) : x(x), y(y), z(z) {}
    };

    template <typename VoxelType>
    class Block : public Vector<VoxelType>
    {
    public:
        using Ptr = std::unique_ptr<Block<VoxelType>>;
        static constexpr size_t kVoxelsPerSide = 8;
        static constexpr size_t kNumVoxels =
            kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;

        Block(MemoryType type);

        VoxelType getVoxel(const Index3D &index, const CudaStream &cuda_stream = CudaStreamOwning()) const;
        void setVoxel(const Index3D &index, const VoxelType value, const CudaStream &cuda_stream = CudaStreamOwning());

        static std::unique_ptr<Block<VoxelType>> copyFrom(const Block<VoxelType> &src, MemoryType target_type,
                                                          const CudaStream &stream = CudaStreamOwning());

    protected:
        void allocateImpl(const CudaStream &cuda_stream = CudaStreamOwning()) override;
        size_t idx(const Index3D &index) const;
    };
} // namespace voxhash