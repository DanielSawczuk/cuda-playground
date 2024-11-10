#include <vector>

enum class MemoryLayout { row_major, col_major };

template <typename dtype_a, typename dtype_b, typename dtype_acc,
          typename dtype_out>
std::vector<dtype_out>
host_gemm_ref(const std::vector<dtype_a> &A, const std::vector<dtype_b> &B,
              size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc,
              MemoryLayout layout_a, MemoryLayout layout_b,
              MemoryLayout layout_c) {
  size_t size_c = layout_c == MemoryLayout::row_major ? M * ldc : N * ldc;

  std::vector<dtype_out> C(size_c, 0.0f);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      dtype_acc acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        dtype_a a = layout_a == MemoryLayout::row_major ? A[i * lda + k]
                                                        : A[k * lda + i];
        dtype_b b = layout_b == MemoryLayout::row_major ? B[k * ldb + j]
                                                        : B[j * ldb + k];
        acc += a * b;
      }
      if (layout_c == MemoryLayout::row_major)
        C[i * ldc + j] = static_cast<dtype_out>(acc);
      else
        C[j * ldc + i] = static_cast<dtype_out>(acc);
    }
  }

  return C;
}
