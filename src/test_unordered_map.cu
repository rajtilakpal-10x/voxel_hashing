
#include <iostream>

#include <thrust/transform.h>

#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE
#include <stdgpu/unordered_map.cuh> // stdgpu::unordered_map

__global__ void init_keys_values(int *keys, int *values, const stdgpu::index_t n)
{
    stdgpu::index_t i = static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    keys[i] = (int)i;
    values[i] = (int)i * i;
}

__global__ void my_insert_kernel(const int *keys, const int *values, const stdgpu::index_t n, stdgpu::unordered_map<int, int> map)
{
    stdgpu::index_t i = static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    map.emplace(keys[i], values[i]);
}

// __global__ void extract_kv(stdgpu::unordered_map<int, int> map,
//                            int *keys,
//                            int *values,
//                            int *count)
// {
//     auto range = map.device_range();

//     for (auto it = range.begin(); it != range.end(); ++it)
//     {
//         // if (!it->occupied || !it->valid)
//         //     continue;

//         int idx = atomicAdd(count, 1);
//         keys[idx] = it->first;
//         values[idx] = it->second;
//     }
// }
struct extract_kv
{
    STDGPU_HOST_DEVICE
    thrust::tuple<int, int>
    operator()(const stdgpu::pair<const int, int> &p) const
    {
        // IMPORTANT: filter invalid entries
        // if (!p.occupied || !p.valid)
        //     return thrust::make_tuple(-1, -1); // sentinel

        return thrust::make_tuple(p.first, p.second);
    }
};

__global__ void insert_one(int key, int value, stdgpu::unordered_map<int, int> map)
{
    map.emplace(key, value);
}

__global__ void setKeys(int *keys, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    keys[i] = i * 5;
}

__global__ void getValues(int *keys, int *values, int n, stdgpu::unordered_map<int, int> map)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int key = keys[i];

    auto it = map.find(key);

    if (it != map.end())
    {
        values[i] = it->second;
    }
    else
    {
        values[i] = -1; // sentinel for "not found"
    }
}

__global__ void getValueSingle(int key,
                               int *out_value,
                               int *out_found,
                               stdgpu::unordered_map<int, int> map)
{
    auto it = map.find(key);

    if (it != map.end())
    {
        *out_value = it->second;
        *out_found = 1;
    }
    else
    {
        *out_found = 0;
    }
}

int main()
{

    // My Implementation
    const stdgpu::index_t n = 100;
    stdgpu::index_t m = 100;

    int *keys{nullptr}, *values{nullptr};
    cudaMalloc(&keys, 2 * m * sizeof(int));
    values = keys + m;

    stdgpu::unordered_map<int, int> map = stdgpu::unordered_map<int, int>::createDeviceObject(n);

    // Insert All

    int tpb = 256;
    int bpg = (m + tpb - 1) / tpb;
    init_keys_values<<<bpg, tpb>>>(keys, values, m);
    cudaDeviceSynchronize();

    my_insert_kernel<<<bpg, tpb>>>(keys, values, m, map);
    cudaDeviceSynchronize();

    // Insert One
    insert_one<<<1, 1>>>(100, 100 * 100, map);
    cudaDeviceSynchronize();

    // Extract All Keys and Values

    int *d_keys{nullptr}, *d_values{nullptr};
    cudaMalloc(&d_keys, 2 * map.size() * sizeof(int));
    d_values = d_keys + map.size();
    int *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    auto range_map = map.device_range();
    thrust::transform(
        range_map.begin(),
        range_map.end(),
        thrust::make_zip_iterator(
            thrust::make_tuple(d_keys, d_values)),
        extract_kv());

    m = map.size();
    int h_keys[m * 2];
    cudaMemcpy(h_keys, d_keys, m * 2 * sizeof(int), cudaMemcpyDefault);
    std::cout << "Keys (" << map.size() << "): ";
    for (int i = 0; i < m; i++)
    {
        std::cout << "(" << h_keys[i] << " " << h_keys[i + m] << "), " << (h_keys[i] * h_keys[i] == h_keys[i + m] ? "valid" : "invalid") << "\n";
    }
    std::cout << "\n";

    // Extract Multiple Values given Keys
    int *g_keys{nullptr}, *g_values{nullptr};
    int num_queries = 10;
    cudaMalloc(&g_keys, 2 * num_queries * sizeof(int));
    g_values = g_keys + num_queries;

    // Key set kernel
    bpg = (num_queries + tpb - 1) / tpb;
    setKeys<<<bpg, tpb>>>(g_keys, num_queries);
    cudaDeviceSynchronize();

    // Value retrieval kernel
    getValues<<<bpg, tpb>>>(g_keys, g_values, num_queries, map);
    cudaDeviceSynchronize();

    cudaMemcpy(h_keys, g_keys, num_queries * 2 * sizeof(int), cudaMemcpyDefault);
    for (int i = 0; i < num_queries; i++)
    {
        std::cout << "(" << h_keys[i] << " " << h_keys[i + num_queries] << "), " << (h_keys[i] * h_keys[i] == h_keys[i + num_queries] ? "valid" : "invalid") << "\n";
    }
    std::cout << "\n";

    // Extract one
    int *d_value, *d_found;
    int value_out = -1;
    cudaMalloc(&d_value, sizeof(int));
    cudaMalloc(&d_found, sizeof(int));

    cudaMemset(d_found, 0, sizeof(int));

    getValueSingle<<<1, 1>>>(100, d_value, d_found, map);
    cudaDeviceSynchronize();

    int h_found;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found)
        cudaMemcpy(&value_out, d_value, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Found: " << (h_found ? "YES" : "NO") << "\n";
    std::cout << "Value: " << value_out << "\n";

    cudaFree(d_value);
    cudaFree(d_found);

    std::cout
        << "Map size: " << map.size() << "\n";

    cudaFree(g_keys);
    cudaFree(d_keys);
    cudaFree(keys);

    stdgpu::unordered_map<int, int>::destroyDeviceObject(map);
}