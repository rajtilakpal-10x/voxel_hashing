#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/block.h"
#include <iostream>
#include <string>

using namespace voxhash;

int main(int argc, char *argv[])
{
    warmupCuda();
    MemoryType type = MemoryType::kDevice;
    Block<float> b(type);
    Index3D index_to_set(0, 0, 1);
    float test_voxel_value = 100;
    b.setVoxel(index_to_set, test_voxel_value);

    Block<float>::Ptr b_h = Block<float>::copyFrom(b, MemoryType::kHost);
    b_h->setVoxel(Index3D(0, 0, 2), 200);

    std::cout << "Data (" << b_h->size() << "): ";
    for (size_t i = 0; i < b_h->size(); i++)
    {
        std::cout << b_h->data()[i] << ",";
    }
    std::cout << "\n";

    float voxel_value = b_h->getVoxel(Index3D(0, 0, 1));
    std::cout << "Voxel Value: " << voxel_value << "\n";

    float *ptr = b.release();
    cudaFree(ptr);
    std::cout << "Valid: " << std::boolalpha << b.valid() << "\n";

    return 0;
}