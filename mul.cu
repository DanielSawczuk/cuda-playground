__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  dest[i] = a[i] * b[i];
}
