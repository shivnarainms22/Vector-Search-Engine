#include "kernels.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 256

// ── Distance kernel ───────────────────────────────────────────────────────────
// One thread per (query, vector) pair. Row-major database (N x D).
// Uncoalesced: consecutive threads access database rows separated by D=128 floats.
__global__ void naive_kernel(
    const float* __restrict__ db,   // (N x D) row-major
    const float* __restrict__ q,    // (Q x D) row-major
    float*       __restrict__ dist, // (Q x N) intermediate distances
    int n, int d
) {
    int vec = blockIdx.x * blockDim.x + threadIdx.x;
    int qry = blockIdx.y;
    if (vec >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = q[qry * d + i] - db[vec * d + i];
        sum += diff * diff;
    }
    dist[qry * n + vec] = sum;
}

// ── Top-K helper ──────────────────────────────────────────────────────────────
// Given Q×N distance matrix, extracts top-K per query using Thrust sort.
// Intentionally duplicated in each kernel TU for isolation — each kernel is
// self-contained and does not share implementation with others.
static void topk_via_thrust(
    const float* d_all_dist,  // (Q x N) device
    float*       d_dist_out,  // (Q x K) device
    int32_t*     d_idx_out,   // (Q x K) device
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

// ── Public launch function ────────────────────────────────────────────────────
void launch_naive(
    const float* d_db_row, const float* d_queries,
    float* d_dist_out, int32_t* d_idx_out,
    int n, int d, int q, int k
) {
    float* d_all = nullptr;
    CUDA_CHECK(cudaMalloc(&d_all, (size_t)q * n * sizeof(float)));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, q);
    naive_kernel<<<grid, block>>>(d_db_row, d_queries, d_all, n, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    topk_via_thrust(d_all, d_dist_out, d_idx_out, n, q, k);

    CUDA_CHECK(cudaFree(d_all));
}
