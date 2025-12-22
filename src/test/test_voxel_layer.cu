#include "voxhash/core/layer.h"
#include <iostream>
#include <string>

using namespace voxhash;

using TsdfBlock = Block<TsdfVoxel>;
using TsdfLayer = VoxelBlockLayer<TsdfBlock>;

__global__ void addWeightKernel(const size_t num_voxels, TsdfVoxel **v)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_voxels)
    {
        v[idx]->weight += 1.0;
    }
}

int main(int argc, char *argv[])
{
    warmupCuda();

    BlockLayerParams p;
    p.block_size = 0.4; // m
    p.memory_type = MemoryType::kDevice;
    p.min_allocated_blocks = 2;
    p.max_allocated_blocks = 5;

    TsdfLayer layer(p);

    Index3D index_to_allocate(10, 1, 2);
    bool allocated = layer.allocateBlock(index_to_allocate);

    std::cout << "Num blocks: " << layer.numBlocks() << " Num voxels: " << layer.numVoxels() << " Voxel size: " << layer.voxel_size() << "\n";

    CudaStreamOwning stream;

    Index3D block_idx(10, 1, 2), voxel_idx(0, 1, 0);
    TsdfVoxel v = layer.getVoxel(block_idx, voxel_idx, stream);
    std::cout << "Tsdf: " << v.tsdf << " Weight: " << v.weight << "\n";

    v.tsdf = 1.0f;
    v.weight = 10.0f;
    layer.storeVoxel(block_idx, voxel_idx, v, stream);

    v = layer.getVoxel(block_idx, voxel_idx, stream);
    std::cout << "Tsdf: " << v.tsdf << " Weight: " << v.weight << "\n";

    // ============

    Vector3f positionL(1.1, 2.1, 3.2);
    getBlockAndVoxelIndexFromPosition(layer.block_size(), positionL, &block_idx, &voxel_idx);
    allocated = layer.allocateBlock(block_idx);

    v = layer.getVoxel(positionL, stream);
    std::cout << "Tsdf: " << v.tsdf << " Weight: " << v.weight << "\n";

    v.tsdf = 2.0f;
    v.weight = 15.0f;
    layer.storeVoxel(positionL, v, stream);

    v = layer.getVoxel(positionL, stream);
    std::cout << "Tsdf: " << v.tsdf << " Weight: " << v.weight << "\n";

    // ===========
    {
        std::vector<Index3D> block_indices;
        std::vector<Index3D> voxel_indices;
        block_indices.push_back(Index3D(0, 0, 1));
        block_indices.push_back(Index3D(-1, 0, 0));
        block_indices.push_back(Index3D(0, -1, 0));
        voxel_indices.push_back(Index3D(2, 0, 1));
        voxel_indices.push_back(Index3D(2, 1, 0));
        voxel_indices.push_back(Index3D(2, 0, 0));
        std::vector<TsdfVoxel> voxels_to_store;
        voxels_to_store.emplace_back(-1.0f, 10.0f);
        voxels_to_store.emplace_back(1.0f, 20.0f);
        voxels_to_store.emplace_back(-1.5f, 30.0f);

        IndexPairs block_and_voxel_indices = {block_indices, voxel_indices};

        Vector<TsdfVoxel>::Ptr voxels_to_store_ = Vector<TsdfVoxel>::copyFrom(voxels_to_store, MemoryType::kHost, stream); // Doesn't matter where we store
        Vector<Bool> stored = layer.storeVoxels(block_and_voxel_indices, *voxels_to_store_, stream);
        Vector<Bool>::Ptr stored_host = Vector<Bool>::copyFrom(stored, MemoryType::kHost, stream);
        stream.synchronize();

        std::cout << "Stored: ";
        for (size_t i = 0; i < stored_host->size(); i++)
            std::cout << (int)((*stored_host)[i]) << ",";
        std::cout << "\n";

        Vector<TsdfVoxel> voxels = layer.getVoxels(block_and_voxel_indices, stream);
        Vector<TsdfVoxel>::Ptr voxels_host = Vector<TsdfVoxel>::copyFrom(voxels, MemoryType::kHost, stream);

        stream.synchronize();

        std::cout << "Voxels: \n";
        for (size_t i = 0; i < voxels_host->size(); i++)
            std::cout << "(" << voxels_host->data()[i].tsdf << "," << voxels_host->data()[i].weight << ")\n";
    }

    // ===========
    {
        std::vector<Vector3f> positions;
        positions.push_back(Vector3f(10.0, 11.0, 12.0));
        positions.push_back(Vector3f(-10.0, 11.0, 12.0));
        positions.push_back(Vector3f(10.0, -11.0, 12.0));
        std::vector<TsdfVoxel> voxels_to_store;
        voxels_to_store.emplace_back(-1.0f, 10.0f);
        voxels_to_store.emplace_back(1.0f, 20.0f);
        voxels_to_store.emplace_back(-1.5f, 30.0f);

        Vector<TsdfVoxel>::Ptr voxels_to_store_ = Vector<TsdfVoxel>::copyFrom(voxels_to_store, MemoryType::kHost, stream); // Doesn't matter where we store
        Vector<Bool> stored = layer.storeVoxels(positions, *voxels_to_store_, stream);
        Vector<Bool>::Ptr stored_host = Vector<Bool>::copyFrom(stored, MemoryType::kHost, stream);
        stream.synchronize();

        std::cout << "Stored: ";
        for (size_t i = 0; i < stored_host->size(); i++)
            std::cout << (int)((*stored_host)[i]) << ",";
        std::cout << "\n";

        Vector<TsdfVoxel> voxels = layer.getVoxels(positions, stream);
        Vector<TsdfVoxel>::Ptr voxels_host = Vector<TsdfVoxel>::copyFrom(voxels, MemoryType::kHost, stream);

        std::cout << "Voxels by Positions: \n";
        for (size_t i = 0; i < voxels_host->size(); i++)
            std::cout << "(" << voxels_host->data()[i].tsdf << "," << voxels_host->data()[i].weight << ")\n";
    }

    // ========
    {
        Index3D block_idx_to_retrieve(0, 0, 1), voxel_idx_to_retrieve(2, 0, 1);
        TsdfVoxel *retrieved_voxel = layer.getVoxelPtr(block_idx_to_retrieve, voxel_idx_to_retrieve, stream);
        TsdfVoxel voxel_on_host(0, 0);
        checkCudaErrors(cudaMemcpyAsync(&voxel_on_host, retrieved_voxel, sizeof(TsdfVoxel), cudaMemcpyDefault, stream));
        stream.synchronize();

        std::cout << "Retrieved Voxel: (" << voxel_on_host.tsdf << "," << voxel_on_host.weight << ")\n";
    }

    // ========
    {
        Vector3f position_to_retrieve(10.0, 11.0, 12.0);
        TsdfVoxel *retrieved_voxel = layer.getVoxelPtr(position_to_retrieve, stream);
        TsdfVoxel voxel_on_host(0, 0);
        checkCudaErrors(cudaMemcpyAsync(&voxel_on_host, retrieved_voxel, sizeof(TsdfVoxel), cudaMemcpyDefault, stream));
        stream.synchronize();

        std::cout << "Retrieved Voxel: (" << voxel_on_host.tsdf << "," << voxel_on_host.weight << ")\n";
    }

    // ========

    {
        std::vector<Index3D> block_indices;
        std::vector<Index3D> voxel_indices;
        block_indices.push_back(Index3D(0, 0, 1));
        block_indices.push_back(Index3D(-1, 0, 0));
        block_indices.push_back(Index3D(0, -1, 0));
        voxel_indices.push_back(Index3D(2, 0, 1));
        voxel_indices.push_back(Index3D(2, 1, 0));
        voxel_indices.push_back(Index3D(2, 0, 0));

        IndexPairs block_and_voxel_indices = {block_indices, voxel_indices};
        Vector<TsdfVoxel *> retrieved_voxels = layer.getVoxelsPtr(block_and_voxel_indices, stream);
        stream.synchronize();

        // Update the retrieved voxels
        if (p.memory_type == MemoryType::kHost)
        {
            for (size_t i = 0; i < retrieved_voxels.size(); i++)
                retrieved_voxels[i]->weight += 1.0;
        }
        else
        {
            constexpr int kNumThreads = 512;
            const int kNumBlocks = voxel_indices.size() / kNumThreads + 1;
            addWeightKernel<<<kNumBlocks, kNumThreads, 0, stream>>>(retrieved_voxels.size(), retrieved_voxels.data());
            stream.synchronize();
            checkCudaErrors(cudaPeekAtLastError());
        }

        Vector<TsdfVoxel> voxels = layer.getVoxels(block_and_voxel_indices, stream);
        Vector<TsdfVoxel>::Ptr voxels_host = Vector<TsdfVoxel>::copyFrom(voxels, MemoryType::kHost, stream);

        stream.synchronize();

        std::cout << "Added Voxels: \n";
        for (size_t i = 0; i < voxels_host->size(); i++)
            std::cout << "(" << voxels_host->data()[i].tsdf << "," << voxels_host->data()[i].weight << ")\n";
    }

    // ========

    {
        std::vector<Vector3f> positions;
        positions.push_back(Vector3f(10.0, 11.0, 12.0));
        positions.push_back(Vector3f(-10.0, 11.0, 12.0));
        positions.push_back(Vector3f(10.0, -11.0, 12.0));

        Vector<TsdfVoxel *> retrieved_voxels = layer.getVoxelsPtr(positions, stream);
        stream.synchronize();

        // Update the retrieved voxels
        if (p.memory_type == MemoryType::kHost)
        {
            for (size_t i = 0; i < retrieved_voxels.size(); i++)
                retrieved_voxels[i]->weight += 1.0;
        }
        else
        {
            constexpr int kNumThreads = 512;
            const int kNumBlocks = positions.size() / kNumThreads + 1;
            addWeightKernel<<<kNumBlocks, kNumThreads, 0, stream>>>(retrieved_voxels.size(), retrieved_voxels.data());
            stream.synchronize();
            checkCudaErrors(cudaPeekAtLastError());
        }

        Vector<TsdfVoxel> voxels = layer.getVoxels(positions, stream);
        Vector<TsdfVoxel>::Ptr voxels_host = Vector<TsdfVoxel>::copyFrom(voxels, MemoryType::kHost, stream);

        stream.synchronize();

        std::cout << "Added Voxels by Position: \n";
        for (size_t i = 0; i < voxels_host->size(); i++)
            std::cout << "(" << voxels_host->data()[i].tsdf << "," << voxels_host->data()[i].weight << ")\n";
    }

    return 0;
}