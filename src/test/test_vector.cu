#include <gtest/gtest.h>
#include <voxhash/core/cuda_utils.h>
#include <voxhash/core/vector.h>

using namespace voxhash;

TEST(TestVector, Create) {
    size_t size = 100;
    MemoryType type = MemoryType::kDevice;

    EXPECT_NO_THROW({ Vector<float> v(size, type); });
}

TEST(TestVector, DataSetAndCopy) {
    EXPECT_NO_THROW({
        size_t size = 100;
        MemoryType type = MemoryType::kDevice;

        Vector<float> v(size, type);
        float test_data = 100;
        CUDA_CHECK(cudaMemcpy(v.data() + 1, &test_data, sizeof(float), cudaMemcpyDefault));

        Vector<float>::Ptr v1 = Vector<float>::copyFrom(v, MemoryType::kHost);

        EXPECT_EQ(v1->data()[1], test_data);
    });
}

TEST(TestVector, DataClear) {
    EXPECT_NO_THROW({
        size_t size = 100;
        MemoryType type = MemoryType::kDevice;

        Vector<float> v(size, type);
        float test_data = 100;
        CUDA_CHECK(cudaMemcpy(v.data() + 1, &test_data, sizeof(float), cudaMemcpyDefault));

        Vector<float>::Ptr v1 = Vector<float>::copyFrom(v, MemoryType::kHost);

        v1->clear();

        EXPECT_NE(v1->data()[1], test_data);
    });
}

TEST(TestVector, SetFrom) {
    EXPECT_NO_THROW({
        size_t size = 100;
        MemoryType type = MemoryType::kDevice;

        Vector<float> v(size, type);
        float test_data = 100;
        CUDA_CHECK(cudaMemcpy(v.data() + 1, &test_data, sizeof(float), cudaMemcpyDefault));

        Vector<float>::Ptr v1 = Vector<float>::copyFrom(v, MemoryType::kHost);

        v1->clear();
        EXPECT_TRUE(v1->setFrom(v));
        EXPECT_EQ(v1->data()[1], test_data);
    });
}

TEST(TestVector, Release) {
    EXPECT_NO_THROW({
        size_t size = 100;
        MemoryType type = MemoryType::kDevice;

        Vector<float> v(size, type);
        float* ptr = v.release();
        EXPECT_FALSE(v.valid());
        CUDA_CHECK(cudaFree(ptr));
    });
}

TEST(TestVector, CopyFromVector) {
    EXPECT_NO_THROW({
        int num_queries = 10;
        std::vector<int> vector_to_convert(num_queries);
        for (size_t i = 0; i < num_queries; i++) {
            vector_to_convert.push_back(i);
        }

        Vector<int>::Ptr v_device = Vector<int>::copyFrom(vector_to_convert, MemoryType::kDevice);
        Vector<int>::Ptr v_host = Vector<int>::copyFrom(*v_device, MemoryType::kHost);

        for (size_t i = 0; i < num_queries; i++) EXPECT_EQ(vector_to_convert[i], (*v_host)[i]);
    });
}