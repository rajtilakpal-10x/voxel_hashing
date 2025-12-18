
#include <iostream>
#include <voxhash/core/cuda_utils.h>

int main(int argc, char *argv[]) {
  std::cout << "Warming up Cuda\n";

  voxhash::warmupCuda();

  std::cout << "Cuda warmed up\n";
}