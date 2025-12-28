
#include "voxhash/core/memory_pool.h"

namespace voxhash {

template <typename BlockType>
BlockMemoryPool<BlockType>::BlockMemoryPool(
        size_t min_allocated_blocks, size_t max_allocated_blocks, MemoryType type)
    : min_allocated_blocks_(min_allocated_blocks),
      max_allocated_blocks_(max_allocated_blocks),
      type_(type) {
    if (min_allocated_blocks_ > 0 && max_allocated_blocks >= min_allocated_blocks) {
        ensureCapacity();
    }
}

template <typename BlockType>
void BlockMemoryPool<BlockType>::ensureCapacity() {
    // Allocate if less than min_allocated_blocks
    int current_size = (int)size();
    int min__ = (int)min_allocated_blocks_;
    if (current_size < min__) {
        int num_blocks_to_allocate = (int)max_allocated_blocks_ - current_size;
        for (size_t i = 0; i < num_blocks_to_allocate; i++) {
            try {
                blocks_.push(std::make_shared<BlockType>(type_));
            } catch (const CudaError& e) {
            }
        }
    }

    // Deallocate if greater than max_allocated_blocks
    if (current_size > max_allocated_blocks_) {
        int num_blocks_to_deallocate = current_size - (int)max_allocated_blocks_;
        for (size_t i = 0; i < num_blocks_to_deallocate; ++i) {
            blocks_.pop();
        }
    }
}

template <typename BlockType>
typename BlockType::Ptr BlockMemoryPool<BlockType>::popBlock() {
    ensureCapacity();
    typename BlockType::Ptr b = blocks_.top();
    blocks_.pop();
    return b;
}

template <typename BlockType>
bool BlockMemoryPool<BlockType>::pushBlock(typename BlockType::Ptr block) {
    if (block->location() != type_) return false;
    block->clear();
    blocks_.push(block);
    ensureCapacity();
    return true;
}

template <typename BlockType>
size_t BlockMemoryPool<BlockType>::size() const {
    return blocks_.size();
}

template <typename BlockType>
BlockMemoryPool<BlockType>::~BlockMemoryPool() {}

template class BlockMemoryPool<Block<int>>;
template class BlockMemoryPool<Block<float>>;
template class BlockMemoryPool<Block<TsdfVoxel>>;
template class BlockMemoryPool<Block<SemanticVoxel>>;

}  // namespace voxhash