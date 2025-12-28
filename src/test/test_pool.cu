#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/memory_pool.h"
#include "voxhash/core/voxels.h"
using namespace std::chrono;

using namespace voxhash;
using TsdfBlock = Block<TsdfVoxel>;

TEST(TestBlockPool, TestBlockPoolSizes) {
    MemoryType type = MemoryType::kDevice;

    size_t min_alloc = 2;
    size_t max_alloc = 5;

    size_t to_pull = 8;
    std::vector<TsdfBlock::Ptr> pulled_blocks;

    {
        BlockMemoryPool<TsdfBlock> pool(min_alloc, max_alloc, type);

        for (size_t i = 0; i < to_pull; i++) {
            pulled_blocks.emplace_back(pool.popBlock());
        }

        EXPECT_EQ(pool.size(), 1);

        for (size_t i = 0; i < pulled_blocks.size(); i++) {
            pool.pushBlock(pulled_blocks[i]);
        }
        EXPECT_EQ(pool.size(), 5);
    }
}
