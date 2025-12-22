
#include "voxhash/core/layer.h"
#include "voxhash/core/block.h"
#include "voxhash/core/voxels.h"

#include <iostream>
namespace voxhash
{

    template <typename BlockType>
    BlockLayer<BlockType>::BlockLayer(const BlockLayerParams &params) : block_size_(params.block_size), pool_(params.min_allocated_blocks, params.max_allocated_blocks, params.memory_type), type_(params.memory_type)
    {
    }

    template <typename BlockType>
    IndexBlockPair<BlockType> BlockLayer<BlockType>::getBlock(const Index3D &index) const
    {
        const auto it = hash_.find(index);
        return std::make_pair(index, it != hash_.end() ? it->second : nullptr);
    }

    template <typename BlockType>
    ConstIndexBlockPairs<BlockType> BlockLayer<BlockType>::getBlocks(const std::vector<Index3D> &indices) const
    {
        std::vector<Index3D> all_indices;
        std::vector<typename BlockType::Ptr> all_blocks;

        for (const auto &index : indices)
        {
            IndexBlockPair<BlockType> block_pair = getBlock(index);
            all_indices.push_back(block_pair.first);
            all_blocks.push_back(block_pair.second);
        }
        return {all_indices, all_blocks};
    }

    template <typename BlockType>
    bool BlockLayer<BlockType>::allocateBlock(const Index3D &index)
    {
        // First, check if aleady a block exists or not
        IndexBlockPair<BlockType> block_pair = getBlock(index);
        if (block_pair.second)
        {
            // If Block already exists, return True
            return true;
        }
        else
        {
            // Add new_block at hash_[index]
            typename BlockType::Ptr new_block = pool_.popBlock();
            if (!new_block)
                return false;
            hash_[index] = new_block;
        }
        return true;
    }

    template <typename BlockType>
    std::vector<bool> BlockLayer<BlockType>::allocateBlocks(const std::vector<Index3D> &indices)
    {
        std::vector<bool> able_to_allocate;
        able_to_allocate.reserve(indices.size());

        for (const auto &index : indices)
            able_to_allocate.push_back(allocateBlock(index));

        return able_to_allocate;
    }

    template <typename BlockType>
    std::vector<Index3D> BlockLayer<BlockType>::getAllBlockIndices() const
    {
        std::vector<Index3D> indices;
        indices.reserve(hash_.size());

        for (const auto &kv : hash_)
        {
            indices.push_back(kv.first);
        }
        return indices;
    }
    template <typename BlockType>
    std::vector<typename BlockType::Ptr> BlockLayer<BlockType>::getAllBlockPointers() const
    {
        std::vector<typename BlockType::Ptr> ptrs;
        ptrs.reserve(hash_.size());

        for (const auto &kv : hash_)
        {
            ptrs.push_back(kv.second);
        }
        return ptrs;
    }

    template <typename BlockType>
    bool BlockLayer<BlockType>::isBlockAllocated(const Index3D &index) const
    {
        const auto it = hash_.find(index);
        return it != hash_.end();
    }

    template <typename BlockType>
    std::vector<bool> BlockLayer<BlockType>::areBlocksAllocated(const std::vector<Index3D> &indices) const
    {
        std::vector<bool> found;
        found.reserve(indices.size());
        for (const auto &index : indices)
            found.push_back(isBlockAllocated(index));
        return found;
    }

    template <typename BlockType>
    bool BlockLayer<BlockType>::storeBlock(const IndexBlockPair<BlockType> &index_block_pair)
    {
        Index3D index = index_block_pair.first;
        typename BlockType::Ptr block_ptr = index_block_pair.second;
        if (!block_ptr)
            return false;

        bool allocation_success = allocateBlock(index);
        if (!allocation_success)
            return false;

        IndexBlockPair<BlockType> layer_index_block_pair = getBlock(index);
        typename BlockType::Ptr layer_block_ptr = layer_index_block_pair.second;
        if (!layer_block_ptr)
            return false;

        layer_block_ptr->setFrom(*block_ptr);
        return true;
    }

    template <typename BlockType>
    std::vector<bool> BlockLayer<BlockType>::storeBlocks(const ConstIndexBlockPairs<BlockType> &index_block_pairs)
    {

        const auto &indices = index_block_pairs.first;
        const auto &blocks = index_block_pairs.second;

        std::vector<bool> stored;
        stored.reserve(indices.size());
        size_t size = indices.size();

        for (size_t i = 0; i < size; i++)
        {
            IndexBlockPair<BlockType> index_block_pair = {indices[i], blocks[i]};
            stored.push_back(storeBlock(index_block_pair));
        }
        return stored;
    }

    template <typename BlockType>
    bool BlockLayer<BlockType>::deAllocateBlock(const Index3D &index)
    {

        auto it = hash_.find(index);
        if (it == hash_.end()) // Nothing to deallocate
            return true;

        bool deallocated = false;
        // Return block to pool
        if (it->second)
            deallocated = pool_.pushBlock(it->second);

        // Remove from hash
        hash_.erase(it);
        return deallocated;
    }

    template <typename BlockType>
    std::vector<bool> BlockLayer<BlockType>::deAllocateBlocks(const std::vector<Index3D> &indices)
    {
        std::vector<bool> deallocated;
        deallocated.reserve(indices.size());

        for (const auto &index : indices)
            deallocated.push_back(deAllocateBlock(index));
        return deallocated;
    }

    template <typename BlockType>
    VoxelBlockLayer<BlockType>::VoxelBlockLayer(const BlockLayerParams &params) : BlockLayer<BlockType>(params) {}

    template <typename BlockType>
    typename BlockType::VoxelType VoxelBlockLayer<BlockType>::getVoxel(const Index3D &block_idx, const Index3D &voxel_idx, const CudaStream &stream) const
    {
        typename BlockType::VoxelType v;
        IndexBlockPair<BlockType> index_block_pair = this->getBlock(block_idx);
        typename BlockType::Ptr block_ptr = index_block_pair.second;
        if (block_ptr)
            v = block_ptr->getVoxel(voxel_idx, stream);
        return v;
    }

    template <typename BlockType>
    typename BlockType::VoxelType VoxelBlockLayer<BlockType>::getVoxel(const Vector3f &positionL, const CudaStream &stream) const
    {
        Index3D block_idx(0, 0, 0), voxel_idx(0, 0, 0);
        getBlockAndVoxelIndexFromPosition(this->block_size_, positionL, &block_idx, &voxel_idx);
        return getVoxel(block_idx, voxel_idx, stream);
    }

    template <typename BlockType>
    bool VoxelBlockLayer<BlockType>::storeVoxel(const Index3D &block_idx, const Index3D &voxel_idx, const typename BlockType::VoxelType voxel, const CudaStream &stream)
    {
        typename BlockType::VoxelType v;
        IndexBlockPair<BlockType> index_block_pair = this->getBlock(block_idx);
        typename BlockType::Ptr block_ptr = index_block_pair.second;
        if (!block_ptr)
            return false;
        block_ptr->setVoxel(voxel_idx, voxel, stream);
        return true; // TODO: Change error check functionality
    }

    template <typename BlockType>
    bool VoxelBlockLayer<BlockType>::storeVoxel(const Vector3f &positionL, const typename BlockType::VoxelType voxel, const CudaStream &stream)
    {
        Index3D block_idx(0, 0, 0), voxel_idx(0, 0, 0);
        getBlockAndVoxelIndexFromPosition(this->block_size_, positionL, &block_idx, &voxel_idx);
        return storeVoxel(block_idx, voxel_idx, voxel, stream);
    }

    template <typename BlockType>
    __global__ void getVoxelsKernel(size_t num_voxels, typename BlockType::VoxelType **block_ptrs, Index3D *voxel_indices, typename BlockType::VoxelType *voxels)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_voxels)
            return;
        if (block_ptrs[idx])
            voxels[idx] = block_ptrs[idx][BlockType::idx(voxel_indices[idx])];
        else
            voxels[idx] = typename BlockType::VoxelType();
    }

    template <typename BlockType>
    Vector<typename BlockType::VoxelType> VoxelBlockLayer<BlockType>::getVoxelsInternal(const Vector<Index3D> &block_indices, const Vector<Index3D> &voxel_indices, const CudaStream &stream) const
    {
        // Collect Block pointers from hash
        Vector<typename BlockType::VoxelType *> block_ptrs_host(block_indices.size(), MemoryType::kHost);
        for (size_t i = 0; i < block_indices.size(); i++)
        {
            typename BlockType::Ptr block_ptr = this->getBlock(block_indices[i]).second;
            if (block_ptr)
                block_ptrs_host[i] = block_ptr.get()->data();
            else
                block_ptrs_host[i] = nullptr;
        }

        // Allocate space for voxels to be retrieved

        Vector<typename BlockType::VoxelType> voxels(voxel_indices.size(), this->type_);

        if (this->type_ == MemoryType::kHost)
        {
            for (size_t i = 0; i < voxel_indices.size(); i++)
            {
                typename BlockType::VoxelType *block_ptr_host = block_ptrs_host[i];
                if (block_ptr_host)
                    voxels[i] = block_ptr_host[BlockType::idx(voxel_indices[i])];
                else
                    voxels[i] = typename BlockType::VoxelType();
            }
        }
        else
        {
            // Send Block pointers to layer location (device or unified)
            typename Vector<typename BlockType::VoxelType *>::Ptr block_ptrs = Vector<typename BlockType::VoxelType *>::copyFrom(block_ptrs_host, this->type_, stream);
            // Send voxel indices to layer location (device or unified)
            Vector<Index3D>::Ptr vi = Vector<Index3D>::copyFrom(voxel_indices, this->type_, stream);

            // Run retrieval kernel
            constexpr int kNumThreads = 512;
            const int kNumBlocks = voxel_indices.size() / kNumThreads + 1;
            getVoxelsKernel<BlockType><<<kNumBlocks, kNumThreads, 0, stream>>>(voxel_indices.size(), block_ptrs->data(), vi->data(), voxels.data());
            stream.synchronize();
            checkCudaErrors(cudaPeekAtLastError());
        }
        return voxels;
    }

    template <typename BlockType>
    Vector<typename BlockType::VoxelType> VoxelBlockLayer<BlockType>::getVoxels(IndexPairs &index_pairs, const CudaStream &stream) const
    {
        const std::vector<Index3D> &block_indices = index_pairs.first;
        const std::vector<Index3D> &voxel_indices = index_pairs.second;

        Vector<Index3D>::Ptr bi = Vector<Index3D>::copyFrom(block_indices, MemoryType::kHost, stream);
        Vector<Index3D>::Ptr vi = Vector<Index3D>::copyFrom(voxel_indices, MemoryType::kHost, stream);
        return getVoxelsInternal(*bi, *vi, stream);
    }

    template <typename BlockType>
    Vector<typename BlockType::VoxelType> VoxelBlockLayer<BlockType>::getVoxels(const std::vector<Vector3f> &positionsL, const CudaStream &stream) const
    {
        Vector<Vector3f>::Ptr p_L = Vector<Vector3f>::copyFrom(positionsL, this->type_, stream);
        auto [block_indices, voxel_indices] = getBlockAndVoxelIndicesFromPositions(this->block_size_, *p_L, stream);

        // Currently, copy indices to host such that CPU-based voxel hashing works
        Vector<Index3D>::Ptr bi = Vector<Index3D>::copyFrom(block_indices, MemoryType::kHost, stream);
        Vector<Index3D>::Ptr vi = Vector<Index3D>::copyFrom(voxel_indices, MemoryType::kHost, stream);
        return getVoxelsInternal(*bi, *vi, stream);
    }

    template <typename BlockType>
    __global__ void storeVoxelsKernel(size_t num_voxels, typename BlockType::VoxelType **block_ptrs, Index3D *voxel_indices, typename BlockType::VoxelType *voxels, Bool *success)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_voxels)
            return;
        if (block_ptrs[idx])
        {
            block_ptrs[idx][BlockType::idx(voxel_indices[idx])] = voxels[idx];
            success[idx] = 1;
        }
        else
        {
            success[idx] = 0;
        }
    }

    template <typename BlockType>
    Vector<Bool> VoxelBlockLayer<BlockType>::storeVoxelsInternal(const Vector<Index3D> &block_indices, const Vector<Index3D> &voxel_indices, const Vector<typename BlockType::VoxelType> &voxels, const CudaStream &stream)
    {

        // Allocate/Collect Block pointers from hash
        Vector<typename BlockType::VoxelType *> block_ptrs_host(block_indices.size(), MemoryType::kHost);
        for (size_t i = 0; i < block_indices.size(); i++)
        {
            bool allocated = this->allocateBlock(block_indices[i]);
            typename BlockType::Ptr block_ptr = this->getBlock(block_indices[i]).second;
            if (allocated || block_ptr)
                block_ptrs_host[i] = block_ptr.get()->data();
            else
                block_ptrs_host[i] = nullptr; // Unable to allocate or find block
        }

        // Transfer the voxels to layer location
        typename Vector<typename BlockType::VoxelType>::Ptr voxels_to_layer = Vector<typename BlockType::VoxelType>::copyFrom(voxels, this->type_, stream);
        Vector<Bool> store_success(voxel_indices.size(), this->type_);

        if (this->type_ == MemoryType::kHost)
        {
            for (size_t i = 0; i < voxel_indices.size(); i++)
            {
                typename BlockType::VoxelType *block_ptr_host = block_ptrs_host[i];
                if (block_ptr_host)
                {
                    block_ptr_host[BlockType::idx(voxel_indices[i])] = (*voxels_to_layer)[i];
                    store_success[i] = 1;
                }
                else
                {
                    store_success[i] = 0;
                }
            }
        }
        else
        {
            // Send Block pointers to layer location (device or unified)
            typename Vector<typename BlockType::VoxelType *>::Ptr block_ptrs = Vector<typename BlockType::VoxelType *>::copyFrom(block_ptrs_host, this->type_, stream);
            // Send voxel indices to layer location (device or unified)
            Vector<Index3D>::Ptr vi = Vector<Index3D>::copyFrom(voxel_indices, this->type_, stream);

            // Run store kernel
            constexpr int kNumThreads = 512;
            const int kNumBlocks = voxel_indices.size() / kNumThreads + 1;
            storeVoxelsKernel<BlockType><<<kNumBlocks, kNumThreads, 0, stream>>>(voxel_indices.size(), block_ptrs->data(), vi->data(), voxels_to_layer->data(), store_success.data());
            stream.synchronize();
            checkCudaErrors(cudaPeekAtLastError());
        }
        return store_success;
    }

    template <typename BlockType>
    Vector<Bool> VoxelBlockLayer<BlockType>::storeVoxels(IndexPairs &index_pairs, const Vector<typename BlockType::VoxelType> &voxels, const CudaStream &stream)
    {
        const std::vector<Index3D> &block_indices = index_pairs.first;
        const std::vector<Index3D> &voxel_indices = index_pairs.second;

        Vector<Index3D>::Ptr bi = Vector<Index3D>::copyFrom(block_indices, MemoryType::kHost, stream);
        Vector<Index3D>::Ptr vi = Vector<Index3D>::copyFrom(voxel_indices, MemoryType::kHost, stream);

        return storeVoxelsInternal(*bi, *vi, voxels, stream);
    }

    template <typename BlockType>
    Vector<Bool> VoxelBlockLayer<BlockType>::storeVoxels(const std::vector<Vector3f> &positionsL, const Vector<typename BlockType::VoxelType> &voxels, const CudaStream &stream)
    {
        Vector<Vector3f>::Ptr p_L = Vector<Vector3f>::copyFrom(positionsL, this->type_, stream);
        auto [block_indices, voxel_indices] = getBlockAndVoxelIndicesFromPositions(this->block_size_, *p_L, stream);

        // Currently, copy indices to host such that CPU-based voxel hashing works
        Vector<Index3D>::Ptr bi = Vector<Index3D>::copyFrom(block_indices, MemoryType::kHost, stream);
        Vector<Index3D>::Ptr vi = Vector<Index3D>::copyFrom(voxel_indices, MemoryType::kHost, stream);

        return storeVoxelsInternal(*bi, *vi, voxels, stream);
    }

    template <typename BlockType>
    typename BlockType::VoxelType *VoxelBlockLayer<BlockType>::getVoxelPtr(const Index3D &block_idx, const Index3D &voxel_idx, const CudaStream &stream) const
    {
        typename BlockType::VoxelType *v = nullptr;
        IndexBlockPair<BlockType> index_block_pair = this->getBlock(block_idx);
        typename BlockType::Ptr block_ptr = index_block_pair.second;
        if (block_ptr)
            v = block_ptr->data() + BlockType::idx(voxel_idx);
        return v;
    }

    template <typename BlockType>
    typename BlockType::VoxelType *VoxelBlockLayer<BlockType>::getVoxelPtr(const Vector3f &positionL, const CudaStream &stream) const
    {
        Index3D block_idx(0, 0, 0), voxel_idx(0, 0, 0);
        getBlockAndVoxelIndexFromPosition(this->block_size_, positionL, &block_idx, &voxel_idx);
        return getVoxelPtr(block_idx, voxel_idx, stream);
    }

    template <typename BlockType>
    Vector<typename BlockType::VoxelType *> VoxelBlockLayer<BlockType>::getVoxelsPtrInternal(const Vector<Index3D> &block_indices, const Vector<Index3D> &voxel_indices, const CudaStream &stream) const
    {
        Vector<typename BlockType::VoxelType *> block_ptrs_host(block_indices.size(), MemoryType::kHost);
        for (size_t i = 0; i < block_indices.size(); i++)
        {
            typename BlockType::Ptr block_ptr = this->getBlock(block_indices[i]).second;
            if (block_ptr)
                block_ptrs_host[i] = block_ptr.get()->data() + BlockType::idx(voxel_indices[i]);
            else
                block_ptrs_host[i] = nullptr; // Unable to allocate or find block
        }
        return std::move(*(Vector<typename BlockType::VoxelType *>::copyFrom(block_ptrs_host, this->type_, stream)));
    }

    template <typename BlockType>
    Vector<typename BlockType::VoxelType *> VoxelBlockLayer<BlockType>::getVoxelsPtr(IndexPairs &index_pairs, const CudaStream &stream) const
    {
        const std::vector<Index3D> &block_indices = index_pairs.first;
        const std::vector<Index3D> &voxel_indices = index_pairs.second;

        Vector<Index3D>::Ptr bi = Vector<Index3D>::copyFrom(block_indices, MemoryType::kHost, stream);
        Vector<Index3D>::Ptr vi = Vector<Index3D>::copyFrom(voxel_indices, MemoryType::kHost, stream);

        return getVoxelsPtrInternal(*bi, *vi, stream);
    }

    template <typename BlockType>
    Vector<typename BlockType::VoxelType *> VoxelBlockLayer<BlockType>::getVoxelsPtr(const std::vector<Vector3f> &positionsL, const CudaStream &stream) const
    {
        Vector<Vector3f>::Ptr p_L = Vector<Vector3f>::copyFrom(positionsL, this->type_, stream);
        auto [block_indices, voxel_indices] = getBlockAndVoxelIndicesFromPositions(this->block_size_, *p_L, stream);

        Vector<Index3D>::Ptr bi = Vector<Index3D>::copyFrom(block_indices, MemoryType::kHost, stream);
        Vector<Index3D>::Ptr vi = Vector<Index3D>::copyFrom(voxel_indices, MemoryType::kHost, stream);

        return getVoxelsPtrInternal(*bi, *vi, stream);
    }

    template class BlockLayer<Block<int>>;
    template class BlockLayer<Block<float>>;
    template class BlockLayer<Block<TsdfVoxel>>;
    template class BlockLayer<Block<SemanticVoxel>>;

    template class VoxelBlockLayer<Block<int>>;
    template class VoxelBlockLayer<Block<float>>;
    template class VoxelBlockLayer<Block<TsdfVoxel>>;
    template class VoxelBlockLayer<Block<SemanticVoxel>>;
}