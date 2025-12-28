
#include <gtest/gtest.h>
#include <voxhash/core/cuda_utils.h>

using namespace voxhash;

TEST(CudaWarmup, WarmupDoesNotThrow) {
    EXPECT_NO_THROW({ warmupCuda(); });
}
