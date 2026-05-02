#include "kernels.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>

#define BLOCK_SIZE  256
#define WARP_SIZE    32
#define N_WARPS     (BLOCK_SIZE / WARP_SIZE)   // 8
#define MAX_K        32

// ── Warp-level min reduction using __shfl_down_sync ──────────────────────────
// After this call, lane 0 holds the minimum (val, idx) across the warp.
__device__ __forceinline__ void warp_reduce_min(float& val, int32_t& idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float   ov = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int32_t oi = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (ov < val) { val = ov; idx = oi; }
    }
}

// ── Warp top-K kernel ─────────────────────────────────────────────────────────
// One block per query.
// Algorithm (K passes, each pass extracts one nearest neighbor):
//   1. Each thread scans its chunk of database vectors, tracking local min.
//   2. __shfl_down_sync reduces 32 threads → warp minimum (lane 0 holds result).
//   3. Warp leaders write 1 candidate each to smem (8 candidates total).
//   4. Thread 0 picks the global minimum from 8 warp candidates.
//   5. Thread 0 records the selected index; all threads skip it next pass.
__global__ void warp_kernel(
    const float*   __restrict__ db,      // (D x N) column-major
    const float*   __restrict__ queries, // (Q x D) row-major
    float*         __restrict__ out_d,   // (Q x K)
    int32_t*       __restrict__ out_i,   // (Q x K)
    int n, int d, int q, int k
) {
    extern __shared__ float s_query[];                   // [D]
    __shared__ float   s_wdist[N_WARPS];                 // warp-level min distances
    __shared__ int32_t s_widx [N_WARPS];                 // warp-level min indices
    __shared__ int32_t s_selected[MAX_K];                // indices chosen so far

    int qry  = blockIdx.x;
    int tid  = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;

    if (qry >= q) return;

    // Load query into smem
    for (int i = tid; i < d; i += blockDim.x)
        s_query[i] = queries[qry * d + i];
    if (tid < k) s_selected[tid] = -1;
    __syncthreads();

    // K passes: each pass finds one nearest neighbor
    for (int ki = 0; ki < k; ++ki) {
        float   local_best     = FLT_MAX;
        int32_t local_best_idx = -1;

        // Each thread scans its strided chunk of database vectors
        for (int v = tid; v < n; v += blockDim.x) {
            // Skip previously selected vectors
            bool skip = false;
            for (int si = 0; si < ki; ++si)
                if (s_selected[si] == v) { skip = true; break; }
            if (skip) continue;

            float dist = 0.0f;
            for (int i = 0; i < d; ++i) {
                float diff = s_query[i] - db[i * n + v];
                dist += diff * diff;
            }
            if (dist < local_best) { local_best = dist; local_best_idx = v; }
        }

        // Warp-level reduction via __shfl_down_sync → lane 0 holds warp minimum
        warp_reduce_min(local_best, local_best_idx);

        // Warp lane 0 writes result to smem
        if (lane == 0) {
            s_wdist[wid] = local_best;
            s_widx [wid] = local_best_idx;
        }
        __syncthreads();

        // Thread 0 picks global winner from N_WARPS warp candidates
        if (tid == 0) {
            float   gmin_d = FLT_MAX;
            int32_t gmin_i = -1;
            for (int w = 0; w < N_WARPS; ++w) {
                if (s_wdist[w] < gmin_d) { gmin_d = s_wdist[w]; gmin_i = s_widx[w]; }
            }
            out_d[qry * k + ki] = gmin_d;
            out_i[qry * k + ki] = gmin_i;
            s_selected[ki]       = gmin_i;
        }
        __syncthreads();
    }
}

void launch_warp(
    const float*   d_db_col,
    const float*   d_queries,
    float*         d_dist_out,
    int32_t*       d_idx_out,
    int n, int d, int q, int k
) {
    if (k > MAX_K)
        throw std::invalid_argument("launch_warp: k must be <= 32");
    if (n <= 0 || q <= 0 || k <= 0 || d <= 0)
        throw std::invalid_argument("launch_warp: invalid arguments");

    size_t smem = (size_t)d * sizeof(float);
    dim3 block(BLOCK_SIZE);
    dim3 grid(q);
    warp_kernel<<<grid, block, smem>>>(
        d_db_col, d_queries, d_dist_out, d_idx_out, n, d, q, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
