
#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/vector.h"
#include <iostream>
#include <string>

using namespace voxhash;

int main(int argc, char *argv[])
{
  warmupCuda();

  size_t size = 100;
  MemoryType type = MemoryType::kDevice;
  Vector<float> v(size, type);
  float test_data = 100;
  cudaMemcpy(v.data() + 1, &test_data, sizeof(float), cudaMemcpyDefault);

  Vector<float>::Ptr v1 = Vector<float>::copyFrom(v, MemoryType::kHost);

  std::cout << "Data (" << v1->size() << "): ";
  for (size_t i = 0; i < v1->size(); i++)
  {
    std::cout << v1->data()[i] << ",";
  }
  std::cout << "\n";

  float *ptr = v.release();
  std::cout << "Valid: " << std::boolalpha << v.valid() << "\n";
  cudaFree(ptr);

  std::cout << "Created vector of size: " << size << " on " << to_string(type) << "\n";
}