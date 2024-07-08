#include <mma.h>
using namespace nvcuda;


const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

extern "C"
__global__ void matmul(half *a, half *b, float *c, int M, int K, int N)
{
    int lda = K;
    int ldb = N;
    int ldc = N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);

    int warp_rank_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int id_m = warp_rank_m * WMMA_M;
    
    int warp_rank_n = blockIdx.y * blockDim.y + threadIdx.y;
    int id_n = warp_rank_n * WMMA_N;
    
#ifdef DEBUG
    printf("[%d][%d] \n", id_m, id_n);
#endif

    for (int i = 0; i < K; i += WMMA_K) {
        wmma::load_matrix_sync(a_frag, a + i + id_m * lda, lda);
        wmma::load_matrix_sync(b_frag, b + i * ldb + id_n, ldb);
    
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(c + id_m * ldc + id_n, acc_frag, ldc, wmma::mem_row_major);
}
