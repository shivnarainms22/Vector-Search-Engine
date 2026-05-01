#include "vector_index.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <stdexcept>

static std::vector<float> make_vectors(int n, int d, unsigned seed = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> v(n * d);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Returns elapsed milliseconds for one search call using CUDA events.
static float time_search(
    VectorIndex& index,
    const float* queries, int q, int k,
    KernelType ktype
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    index.search(queries, q, k, ktype);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

struct PerfTest {
    KernelType  type;
    const char* name;
    float       max_ms;
};

int main() {
    const int N = 100'000, D = 128, Q = 10, K = 10;

    auto db      = make_vectors(N, D, 42);
    auto queries = make_vectors(Q, D, 99);

    VectorIndex index(D);
    index.add(db.data(), N);

    // Warmup run (first kernel launch pays JIT cost)
    index.search(queries.data(), Q, K, KernelType::NAIVE);

    PerfTest tests[] = {
        {KernelType::NAIVE,     "naive",     5000.0f},
        {KernelType::SMEM,      "smem",      2000.0f},
        {KernelType::COALESCED, "coalesced",  500.0f},
        {KernelType::WARP,      "warp",       500.0f},
    };

    std::cout << "Performance (N=" << N << ", D=" << D
              << ", Q=" << Q << ", K=" << K << ")\n";
    std::cout << "  Kernel       Time(ms)   Budget   Status\n";

    bool all_pass = true;
    for (auto& t : tests) {
        float ms = time_search(index, queries.data(), Q, K, t.type);
        bool  ok = ms < t.max_ms;
        std::cout << "  " << t.name
                  << "\t\t" << ms
                  << "\t<" << t.max_ms
                  << "\t[" << (ok ? "PASS" : "FAIL") << "]\n";
        all_pass &= ok;
    }

    std::cout << (all_pass ? "\nAll perf tests PASSED\n" : "\nSome perf tests FAILED\n");
    return all_pass ? 0 : 1;
}
