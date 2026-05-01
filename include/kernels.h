#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// All launch_* functions: input queries are row-major (Q x D).
// Output: d_dist_out (Q x K float32), d_idx_out (Q x K int32_t).

void launch_naive(
    const float*   d_db_row,   // (N x D) row-major
    const float*   d_queries,  // (Q x D) row-major
    float*         d_dist_out, // (Q x K)
    int32_t*       d_idx_out,  // (Q x K)
    int n, int d, int q, int k
);

void launch_smem(
    const float*   d_db_row,
    const float*   d_queries,
    float*         d_dist_out,
    int32_t*       d_idx_out,
    int n, int d, int q, int k
);

void launch_coalesced(
    const float*   d_db_col,   // (D x N) column-major
    const float*   d_queries,
    float*         d_dist_out,
    int32_t*       d_idx_out,
    int n, int d, int q, int k
);

void launch_warp(
    const float*   d_db_col,   // (D x N) column-major
    const float*   d_queries,
    float*         d_dist_out,
    int32_t*       d_idx_out,
    int n, int d, int q, int k
);
