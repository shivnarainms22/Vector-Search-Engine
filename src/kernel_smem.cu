#include "kernels.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 256

// ── Top-K helper ──────────────────────────────────────────────────────────────
// Intentionally duplicated from kernel_naive.cu — each kernel TU is self-contained.
static void topk_via_thrust(
    const float* d_all_dist,
    float*       d_dist_out,
    int32_t*     d_idx_out,
    int n, int q, int k
) {
    thrust::device_vector<float>   tmp_dist(n);
    thrust::device_vector<int32_t> tmp_idx(n);

    for (int qi = 0; qi < q; ++qi) {
        thrust::copy(thrust::device,
                     d_all_dist + qi * n,
                     d_all_dist + qi * n + n,
                     tmp_dist.begin());
        thrust::sequence(thrust::device, tmp_idx.begin(), tmp_idx.end());
        thrust::sort_by_key(thrust::device,
                            tmp_dist.begin(), tmp_dist.end(),
                            tmp_idx.begin());

        CUDA_CHECK(cudaMemcpy(d_dist_out + qi * k,
                              thrust::raw_pointer_cast(tmp_dist.data()),
                              k * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_idx_out + qi * k,
                              thrust::raw_pointer_cast(tmp_idx.data()),
                              k * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    }
}

// ── Distance kernel ───────────────────────────────────────────────────────────
// Optimization over naive: query vector cached in shared memory.
// All BLOCK_SIZE threads read query dims from smem (1 broadcast load per dim)
// instead of each thread reading D floats from global memory independently.
// Database is still row-major — database accesses remain uncoalesced.
__global__ void smem_kernel(
    const float* __restrict__ db,   // (N x D) row-major
    const float* __restrict__ q,    // (Q x D) row-major
    float*       __restrict__ dist, // (Q x N)
    int n, int d
) {
    extern __shared__ float s_query[];  // [D] — one query cached per block

    int vec = blockIdx.x * blockDim.x + threadIdx.x;
    int qry = blockIdx.y;

    // Cooperatively load this query into shared memory
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        s_query[i] = q[qry * d + i];
    __syncthreads();

    if (vec >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = s_query[i] - db[vec * d + i];  // db access still uncoalesced
        sum += diff * diff;
    }
    dist[qry * n + vec] = sum;
}

// ── Public launch function ────────────────────────────────────────────────────
void launch_smem(
    const float* d_db_row, const float* d_queries,
    float* d_dist_out, int32_t* d_idx_out,
    int n, int d, int q, int k
) {
    if (k > n || n <= 0 || q <= 0 || k <= 0 || d <= 0)
        throw std::invalid_argument("launch_smem: invalid arguments");

    float* d_all = nullptr;
    CUDA_CHECK(cudaMalloc(&d_all, (size_t)q * n * sizeof(float)));

    size_t smem = (size_t)d * sizeof(float);
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, q);
    smem_kernel<<<grid, block, smem>>>(d_db_row, d_queries, d_all, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    topk_via_thrust(d_all, d_dist_out, d_idx_out, n, q, k);

    CUDA_CHECK(cudaFree(d_all));
}
