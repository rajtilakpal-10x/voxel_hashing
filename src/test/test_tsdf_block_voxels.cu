#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/block.h"
#include "voxhash/core/voxels.h"
#include <iostream>
#include <string>

using namespace voxhash;

int main(int argc, char *argv[])
{
    warmupCuda();

    MemoryType type = MemoryType::kDevice;
    size_t size = 10;
    Index3D index_to_set(0, 0, 1);
    TsdfVoxel test_voxel_value = {1.0f, 10.0f};
    Vector<TsdfVoxel> v(size, type);
    cudaMemcpy(v.data() + 1, &test_voxel_value, sizeof(TsdfVoxel), cudaMemcpyDefault);

    Vector<TsdfVoxel>::Ptr v1 = Vector<TsdfVoxel>::copyFrom(v, MemoryType::kHost);

    std::cout << "Data (" << v1->size() << "): ";
    for (size_t i = 0; i < v1->size(); i++)
    {
        std::cout << "(" << v1->data()[i].tsdf << "," << v1->data()[i].weight << "), ";
    }
    std::cout << "\n";

    Block<TsdfVoxel> b(type);
    b.setVoxel(index_to_set, test_voxel_value);

    Block<TsdfVoxel>::Ptr b_h = Block<TsdfVoxel>::copyFrom(b, MemoryType::kHost);
    b_h->setVoxel(Index3D(0, 0, 2), TsdfVoxel{2.0f, 5.0f});

    std::cout << "Data (" << b_h->size() << "): ";
    for (size_t i = 0; i < b_h->size(); i++)
    {
        std::cout << "(" << b_h->data()[i].tsdf << "," << b_h->data()[i].weight << "), ";
    }
    std::cout << "\n";

    TsdfVoxel voxel_value_retrieved = b_h->getVoxel(Index3D(0, 0, 1));
    std::cout << "Voxel Value: (" << voxel_value_retrieved.tsdf << "," << voxel_value_retrieved.weight << ")\n";

    TsdfVoxel *ptr = b.release();
    cudaFree(ptr);
    std::cout << "Valid: " << std::boolalpha << b.valid() << "\n";

    return 0;
}