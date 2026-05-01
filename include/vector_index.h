#pragma once
#include <cstdint>
#include <vector>

enum class KernelType { NAIVE, SMEM, COALESCED, WARP };

struct SearchResult {
    std::vector<float>   distances;  // Q*K floats, row-major
    std::vector<int32_t> indices;    // Q*K int32_t, row-major
};

class VectorIndex {
public:
    explicit VectorIndex(int dim);
    ~VectorIndex();

    // Add N vectors from host memory, row-major layout (N x D)
    void add(const float* vectors, int n);

    // GPU search: Q queries (Q x D, row-major), return top-K per query
    SearchResult search(const float* queries, int q, int k, KernelType kernel) const;

    // CPU brute-force reference (used for correctness verification)
    SearchResult search_cpu(const float* queries, int q, int k) const;

    int dim() const { return dim_; }
    int n()   const { return n_;   }

private:
    int    dim_;
    int    n_     = 0;
    float* h_db_  = nullptr;  // pinned host, row-major (N x D)
    float* d_row_ = nullptr;  // device row-major (N x D)  — kernels 1, 2
    float* d_col_ = nullptr;  // device column-major (D x N) — kernels 3, 4

    void transpose_to_col_major();
};
