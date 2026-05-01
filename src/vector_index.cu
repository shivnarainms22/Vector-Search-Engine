#include "vector_index.h"
#include "kernels.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

// ── Transpose kernel: row-major (N x D) → column-major (D x N) ──────────────
__global__ void transpose_kernel(
    const float* __restrict__ row,
    float*       __restrict__ col,
    int n, int d
) {
    int v   = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockIdx.y * blockDim.y + threadIdx.y;
    if (v >= n || dim >= d) return;
    col[dim * n + v] = row[v * d + dim];
}

// ── Constructor / Destructor ──────────────────────────────────────────────────
VectorIndex::VectorIndex(int dim) : dim_(dim) {}

VectorIndex::~VectorIndex() {
    if (h_db_)  cudaFreeHost(h_db_);
    if (d_row_) cudaFree(d_row_);
    if (d_col_) cudaFree(d_col_);
}

// ── add ───────────────────────────────────────────────────────────────────────
void VectorIndex::add(const float* vectors, int n) {
    if (n_ > 0) throw std::runtime_error("add() called twice; re-create the index");
    if (!vectors || n <= 0) throw std::invalid_argument("add: vectors must be non-null and n > 0");
    n_ = n;

    size_t bytes = (size_t)n * dim_ * sizeof(float);

    CUDA_CHECK(cudaMallocHost(&h_db_, bytes));
    std::memcpy(h_db_, vectors, bytes);

    CUDA_CHECK(cudaMalloc(&d_row_, bytes));
    CUDA_CHECK(cudaMemcpy(d_row_, h_db_, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_col_, bytes));
    transpose_to_col_major();
}

void VectorIndex::transpose_to_col_major() {
    dim3 block(32, 32);
    dim3 grid((n_ + 31) / 32, (dim_ + 31) / 32);
    transpose_kernel<<<grid, block>>>(d_row_, d_col_, n_, dim_);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── search ────────────────────────────────────────────────────────────────────
SearchResult VectorIndex::search(
    const float* queries, int q, int k, KernelType kernel
) const {
    if (n_ == 0) throw std::runtime_error("search: index is empty, call add() first");
    if (k > n_) throw std::invalid_argument("k > n");

    float*   d_q;
    float*   d_dist;
    int32_t* d_idx;

    CUDA_CHECK(cudaMalloc(&d_q,    (size_t)q * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)q * k    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx,  (size_t)q * k    * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_q, queries, (size_t)q * dim_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    SearchResult result;
    try {
        switch (kernel) {
            case KernelType::NAIVE:
                launch_naive(d_row_, d_q, d_dist, d_idx, n_, dim_, q, k);
                break;
            case KernelType::SMEM:
                launch_smem(d_row_, d_q, d_dist, d_idx, n_, dim_, q, k);
                break;
            case KernelType::COALESCED:
                launch_coalesced(d_col_, d_q, d_dist, d_idx, n_, dim_, q, k);
                break;
            case KernelType::WARP:
                launch_warp(d_col_, d_q, d_dist, d_idx, n_, dim_, q, k);
                break;
            default:
                throw std::invalid_argument("search: unknown KernelType");
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        result.distances.resize(q * k);
        result.indices.resize(q * k);
        CUDA_CHECK(cudaMemcpy(result.distances.data(), d_dist,
                              (size_t)q * k * sizeof(float),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result.indices.data(),   d_idx,
                              (size_t)q * k * sizeof(int32_t), cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_q);
        cudaFree(d_dist);
        cudaFree(d_idx);
        throw;
    }
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_idx));

    return result;
}

// ── search_cpu (reference) ────────────────────────────────────────────────────
SearchResult VectorIndex::search_cpu(
    const float* queries, int q, int k
) const {
    if (n_ == 0) throw std::runtime_error("search_cpu: index is empty, call add() first");
    SearchResult result;
    result.distances.resize(q * k);
    result.indices.resize(q * k);

    std::vector<float>   dists(n_);
    std::vector<int32_t> order(n_);

    for (int qi = 0; qi < q; ++qi) {
        for (int vi = 0; vi < n_; ++vi) {
            float sum = 0.0f;
            for (int di = 0; di < dim_; ++di) {
                float diff = queries[qi * dim_ + di] - h_db_[vi * dim_ + di];
                sum += diff * diff;
            }
            dists[vi]  = sum;
            order[vi]  = vi;
        }
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int a, int b) { return dists[a] < dists[b]; });
        for (int ki = 0; ki < k; ++ki) {
            result.indices[qi * k + ki]   = order[ki];
            result.distances[qi * k + ki] = dists[order[ki]];
        }
    }
    return result;
}
