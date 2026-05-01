#include "vector_index.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// ── Helpers ───────────────────────────────────────────────────────────────────
static std::vector<float> make_random_vectors(int n, int d, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n * d);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Returns true if the top-K distances from `got` match `ref` within tolerance.
static bool top_k_match(
    const SearchResult& ref, const SearchResult& got,
    int q, int k, float tol = 1e-3f
) {
    for (int qi = 0; qi < q; ++qi) {
        for (int ki = 0; ki < k; ++ki) {
            float dr = ref.distances[qi * k + ki];
            float dg = got.distances[qi * k + ki];
            float rel_err = std::abs(dr - dg) / (1.0f + std::abs(dr));
            if (rel_err > tol) {
                std::cerr << "  Distance mismatch q=" << qi << " k=" << ki
                          << " ref=" << dr << " got=" << dg
                          << " rel_err=" << rel_err << "\n";
                return false;
            }
        }
    }
    return true;
}

static bool run_test(
    VectorIndex& index,
    const float* queries, int q, int k,
    const SearchResult& ref,
    KernelType ktype, const char* name
) {
    SearchResult got = index.search(queries, q, k, ktype);
    bool ok = top_k_match(ref, got, q, k);
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name << "\n";
    return ok;
}

// ── Tests ─────────────────────────────────────────────────────────────────────
static bool test_basic_correctness() {
    std::cout << "test_basic_correctness (N=1000, D=128, Q=10, K=10)\n";
    const int N = 1000, D = 128, Q = 10, K = 10;

    auto db      = make_random_vectors(N, D, 42);
    auto queries = make_random_vectors(Q, D, 99);

    VectorIndex index(D);
    index.add(db.data(), N);
    SearchResult ref = index.search_cpu(queries.data(), Q, K);

    bool all_pass = true;
    all_pass &= run_test(index, queries.data(), Q, K, ref, KernelType::NAIVE,     "naive");
    all_pass &= run_test(index, queries.data(), Q, K, ref, KernelType::SMEM,      "smem");
    all_pass &= run_test(index, queries.data(), Q, K, ref, KernelType::COALESCED, "coalesced");
    all_pass &= run_test(index, queries.data(), Q, K, ref, KernelType::WARP,      "warp");
    return all_pass;
}

static bool test_k_gt_n_throws() {
    std::cout << "test_k_gt_n_throws\n";
    auto db = make_random_vectors(10, 128, 1);
    float q[128] = {};
    VectorIndex index(128);
    index.add(db.data(), 10);
    try {
        index.search(q, 1, 20, KernelType::NAIVE);
        std::cerr << "  [FAIL] expected exception not thrown\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "  [PASS] k_gt_n_throws\n";
        return true;
    }
}

static bool test_k1(KernelType ktype, const char* name) {
    std::cout << "test_k1 " << name << "\n";
    const int N = 500, D = 64, Q = 5;

    auto db      = make_random_vectors(N, D, 7);
    auto queries = make_random_vectors(Q, D, 8);

    VectorIndex index(D);
    index.add(db.data(), N);

    SearchResult ref = index.search_cpu(queries.data(), Q, 1);
    SearchResult got = index.search(queries.data(), Q, 1, ktype);
    bool ok = top_k_match(ref, got, Q, 1);
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "]\n";
    return ok;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    bool all = true;
    all &= test_basic_correctness();
    all &= test_k_gt_n_throws();
    all &= test_k1(KernelType::NAIVE,     "naive");
    all &= test_k1(KernelType::SMEM,      "smem");
    all &= test_k1(KernelType::COALESCED, "coalesced");
    all &= test_k1(KernelType::WARP,      "warp");

    std::cout << (all ? "\nAll tests PASSED\n" : "\nSome tests FAILED\n");
    return all ? 0 : 1;
}
