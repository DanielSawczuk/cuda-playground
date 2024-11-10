#include "cuda_intellisense.hpp"

extern "C" __global__ void gemm(float *A, float *B, float *C, int M, int N,
                                int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  float oob = (m >= M) || (n >= N);
  if (oob)
    return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[m * K + k] * B[k * N + n];
  }
  C[m * N + n] = acc;
}
