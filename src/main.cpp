#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include <cuda.h>

#include "utils/reference.hpp"
#include "utils/utils.hpp"

int main(int argc, char **argv) {
  uint32_t M = 1024;
  uint32_t N = 1024;
  uint32_t K = 1024;
  using dtype = float;
  const int num_iterations = 25;

  if (argc != 2) {
    std::cerr
        << "Provide the path to the kernel PTX as an argument to the program!"
        << std::endl;
    exit(EXIT_FAILURE);
  }

  size_t size_A = M * K * sizeof(dtype);
  size_t size_B = K * N * sizeof(dtype);
  size_t size_C = M * N * sizeof(dtype);
  std::vector<dtype> h_A = generate_random_vector<dtype>(M * K, -1.0, 1.0);
  std::vector<dtype> h_B = generate_random_vector<dtype>(K * N, -1.0, 1.0);
  std::vector<dtype> h_C(M * N, 0.0f);

  CUDA_CHECK(cuInit(0));

  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));

  CUcontext context;
  CUDA_CHECK(cuCtxCreate(&context, 0, device));

  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  CUmodule module;
  CUDA_CHECK(cuModuleLoad(&module, argv[1]));

  CUfunction kernel;
  CUDA_CHECK(cuModuleGetFunction(&kernel, module, "gemm"));

  CUdeviceptr d_A, d_B, d_C;
  CUDA_CHECK(cuMemAlloc(&d_A, size_A));
  CUDA_CHECK(cuMemAlloc(&d_B, size_B));
  CUDA_CHECK(cuMemAlloc(&d_C, size_C));

  CUDA_CHECK(cuMemcpyHtoDAsync(d_A, h_A.data(), size_A, stream));
  CUDA_CHECK(cuMemcpyHtoDAsync(d_B, h_B.data(), size_B, stream));
  CUDA_CHECK(cuMemcpyHtoDAsync(d_C, h_C.data(), size_C, stream));

  uint32_t threads_per_block[3] = {16, 16, 1};
  uint32_t blocks_per_grid[3] = {div_up(N, threads_per_block[0]),
                                 div_up(M, threads_per_block[1]), 1};

  std::cout << "Threads per block: (" << threads_per_block[0] << ", "
            << threads_per_block[1] << ", " << threads_per_block[2] << ")\n";
  std::cout << "Blocks per grid: (" << blocks_per_grid[0] << ", "
            << blocks_per_grid[1] << ", " << blocks_per_grid[2] << ")\n";

  std::vector<CUevent> start_events(num_iterations);
  std::vector<CUevent> stop_events(num_iterations);
  for (int i = 0; i < num_iterations; ++i) {
    CUDA_CHECK(cuEventCreate(&start_events[i], CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&stop_events[i], CU_EVENT_DEFAULT));
  }

  auto kernel_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    CUDA_CHECK(cuEventRecord(start_events[i], stream));
    void *args[] = {&d_A, &d_B, &d_C, &M, &N, &K};
    CUDA_CHECK(cuLaunchKernel(kernel, blocks_per_grid[0], blocks_per_grid[1],
                              blocks_per_grid[2], threads_per_block[0],
                              threads_per_block[1], threads_per_block[2], 0,
                              stream, args, 0));
    CUDA_CHECK(cuEventRecord(stop_events[i], stream));
  }
  CUDA_CHECK(cuStreamSynchronize(stream));
  auto kernel_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> kernel_duration =
      kernel_end - kernel_start;

  std::vector<float> times(num_iterations);
  for (int i = 0; i < num_iterations; ++i) {
    CUDA_CHECK(cuEventElapsedTime(&times[i], start_events[i], stop_events[i]));

    CUDA_CHECK(cuEventDestroy(start_events[i]));
    CUDA_CHECK(cuEventDestroy(stop_events[i]));
  }

  CUDA_CHECK(cuMemcpyDtoHAsync(h_C.data(), d_C, size_C, stream));

  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  float average_time = sum / num_iterations;
  std::sort(times.begin(), times.end());
  float median_time = times[num_iterations / 2];
  float max_time = times[num_iterations - 1];
  float min_time = times[0];

  std::cout << "[GPU] Average time: " << average_time << " ms\n";
  std::cout << "[GPU] Median time: " << median_time << " ms\n";
  std::cout << "[GPU] Minimum time: " << min_time << " ms\n";
  std::cout << "[GPU] Maximum time: " << max_time << " ms\n";

  std::cout << "[Wallclock] average time: "
            << kernel_duration.count() / num_iterations << " ms\n";

  std::cout << "Calculating reference..." << std::endl;
  auto h_C_ref = host_gemm_ref<dtype, dtype, float, dtype>(
      h_A, h_B, M, N, K, K, N, N, MemoryLayout::row_major,
      MemoryLayout::row_major, MemoryLayout::row_major);
  const double atol = 1e-5;
  const double rtol = 1e-4;
  bool passed = compare_tensors(h_C, h_C_ref, atol, rtol);

  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

  CUDA_CHECK(cuMemFree(d_A));
  CUDA_CHECK(cuMemFree(d_B));
  CUDA_CHECK(cuMemFree(d_C));
  CUDA_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cuStreamDestroy(stream));
  CUDA_CHECK(cuCtxDestroy(context));

  return 0;
}
