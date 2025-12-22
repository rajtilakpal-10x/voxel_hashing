#pragma once

#include "voxhash/core/hash.h"
#include "voxhash/core/memory_pool.h"
#include "voxhash/core/indexing.h"
#include <memory>

namespace voxhash
{

    struct BlockLayerParams
    {
        float block_size;            // Size of each block in m
        MemoryType memory_type;      // Where to store the blocks of this layer
        size_t min_allocated_blocks; // Minimum number of pre-allocated blocks to hold
                                     // for use
        size_t max_allocated_blocks; // Maximum number of pre-allocated blocks to hold
                                     // for use
    };

    class BaseLayer
    {
    public:
        typedef std::shared_ptr<BaseLayer> Ptr;
        typedef std::shared_ptr<const BaseLayer> ConstPtr;

        virtual ~BaseLayer() = default;
    };

    template <typename BlockType>
    class BlockLayer : public BaseLayer
    { // TODO: Move to GPU-based implementations of CPU hashmap if necessary to parallellize bulk calls for blocks
    public:
        using Ptr = std::shared_ptr<BlockLayer<BlockType>>;

        BlockLayer() = delete;
        BlockLayer(const BlockLayerParams &params);
        virtual ~BlockLayer() {}

        virtual size_t numBlocks() const
        {
            return hash_.size();
        }
        virtual size_t numPreAllocatedBlocksInPool() const { return pool_.size(); }
        virtual float block_size() const { return block_size_; }
        virtual MemoryType location() const { return type_; }
        virtual IndexBlockPair<BlockType> getBlock(const Index3D &index) const;                       // Get Block at index (returns nullptr if not found)
        virtual ConstIndexBlockPairs<BlockType> getBlocks(const std::vector<Index3D> &indices) const; // Get Blocks corresponding to indices

        virtual bool storeBlock(const IndexBlockPair<BlockType> &index_block_pair);                      // Store block at given index (allocates if not allocated)
        virtual std::vector<bool> storeBlocks(const ConstIndexBlockPairs<BlockType> &index_block_pairs); // Store blocks at given indices (allocates if not allocated)

        virtual std::vector<Index3D> getAllBlockIndices() const;
        virtual std::vector<typename BlockType::Ptr> getAllBlockPointers() const;

        virtual bool isBlockAllocated(const Index3D &index) const;
        virtual std::vector<bool> areBlocksAllocated(const std::vector<Index3D> &indices) const;

        virtual bool allocateBlock(const Index3D &index); // Allocates a new block at index. Returns False if could not allocate. Ignores allocation and returns True if already allocated
        virtual std::vector<bool> allocateBlocks(const std::vector<Index3D> &indices);

        virtual bool deAllocateBlock(const Index3D &index);
        virtual std::vector<bool> deAllocateBlocks(const std::vector<Index3D> &indices);

    protected:
        float block_size_{0.0f};
        BlockMemoryPool<BlockType> pool_;
        Index3DHashMapType<BlockType> hash_;
        MemoryType type_{MemoryType::kHost};
    };

    template <typename BlockType>
    class VoxelBlockLayer : public BlockLayer<BlockType>
    {
    public:
        using Ptr = std::shared_ptr<VoxelBlockLayer<BlockType>>;
        VoxelBlockLayer() = delete;

        VoxelBlockLayer(const BlockLayerParams &params);

        virtual size_t numVoxels() const
        {
            return this->numBlocks() * BlockType::kNumVoxels;
        }
        virtual float voxel_size() const
        {
            return blockSizeToVoxelSize(this->block_size_);
        }

        virtual typename BlockType::VoxelType getVoxel(const Index3D &block_idx, const Index3D &voxel_idx, const CudaStream &stream = CudaStreamOwning()) const;
        virtual typename BlockType::VoxelType getVoxel(const Vector3f &positionL, const CudaStream &stream = CudaStreamOwning()) const;
        virtual Vector<typename BlockType::VoxelType> getVoxels(IndexPairs &index_pairs, const CudaStream &stream = CudaStreamOwning()) const; // Returns Vector of voxels in the layer location
        virtual Vector<typename BlockType::VoxelType> getVoxels(const std::vector<Vector3f> &positionsL, const CudaStream &stream = CudaStreamOwning()) const;

        virtual typename BlockType::VoxelType *getVoxelPtr(const Index3D &block_idx, const Index3D &voxel_idx, const CudaStream &stream = CudaStreamOwning()) const;
        virtual typename BlockType::VoxelType *getVoxelPtr(const Vector3f &positionL, const CudaStream &stream = CudaStreamOwning()) const;
        virtual Vector<typename BlockType::VoxelType *> getVoxelsPtr(IndexPairs &index_pairs, const CudaStream &stream = CudaStreamOwning()) const;
        virtual Vector<typename BlockType::VoxelType *> getVoxelsPtr(const std::vector<Vector3f> &positionsL, const CudaStream &stream = CudaStreamOwning()) const;

        virtual bool storeVoxel(const Index3D &block_idx, const Index3D &voxel_idx, const typename BlockType::VoxelType voxel, const CudaStream &stream = CudaStreamOwning());
        virtual bool storeVoxel(const Vector3f &positionL, const typename BlockType::VoxelType voxel, const CudaStream &stream = CudaStreamOwning());
        virtual Vector<Bool> storeVoxels(IndexPairs &index_pairs, const Vector<typename BlockType::VoxelType> &voxels, const CudaStream &stream = CudaStreamOwning());
        virtual Vector<Bool> storeVoxels(const std::vector<Vector3f> &positionsL, const Vector<typename BlockType::VoxelType> &voxels, const CudaStream &stream = CudaStreamOwning());

    protected:
        virtual Vector<typename BlockType::VoxelType> getVoxelsInternal(const Vector<Index3D> &block_indices, const Vector<Index3D> &voxel_indices, const CudaStream &stream) const;
        virtual Vector<typename BlockType::VoxelType *> getVoxelsPtrInternal(const Vector<Index3D> &block_indices, const Vector<Index3D> &voxel_indices, const CudaStream &stream) const;
        virtual Vector<Bool> storeVoxelsInternal(const Vector<Index3D> &block_indices, const Vector<Index3D> &voxel_indices, const Vector<typename BlockType::VoxelType> &voxels, const CudaStream &stream);
    };

} // namespace voxhash