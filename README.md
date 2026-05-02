# CUDA Vector Search Engine (FAISS-lite)

GPU-accelerated nearest-neighbor search with progressively optimized CUDA kernels, benchmarked on A100 SXM4 40GB.

## Overview

Given a dataset of N float32 vectors and Q queries, computes Top-K nearest neighbors by L2 distance.
Four CUDA kernels each add one optimization over the previous, producing a measurable speedup at every step.

## Build

```bash
# Install dependencies
pip install pybind11 numpy pytest

# Configure and build (on RunPod / Linux with CUDA 12.x)
cmake -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build --parallel
```

## Test

```bash
# C++ correctness tests (all 4 kernels vs CPU reference)
./build/test_correctness

# C++ performance regression guard
./build/test_perf

# Python smoke tests
python -m pytest tests/test_bindings.py -v
```

## Benchmark

```bash
# Download SIFT1M (~560MB)
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar -xzf sift.tar.gz

PYTHONPATH=build python python/benchmark.py --data-dir sift/
```

## Benchmark Results

*Benchmarked on A100 SXM4 40GB via RunPod. N=1M, D=128, Q=100, K=10.*

| Version        | Time (ms) | Speedup  | Recall@10 |
|----------------|-----------|----------|-----------|
| CPU baseline   | 21,415    | 1×       | 100%      |
| CUDA naive     | 142       | 151×     | 100%      |
| CUDA smem      | 140       | 153×     | 100%      |
| CUDA coalesced | 86        | 250×     | 100%      |
| CUDA warp      | 461       | 46×      | 100%      |

**Notes:**
- SMEM vs Naive: the query vector (128 floats = 512 bytes) fits in L1 cache on A100, so the shared memory load adds overhead without reducing traffic.
- Warp kernel: uses K sequential passes over N vectors instead of a single pass + Thrust sort. At K=10 and N=1M that is 10× more distance computations, which outweighs the savings from eliminating the Q×N intermediate buffer. This tradeoff favors the warp approach only at very large N or very tight memory constraints.

## Optimization Story

Each kernel adds exactly one optimization over the previous:

| Kernel    | Optimization                                    | Key technique               |
|-----------|-------------------------------------------------|-----------------------------|
| Naive     | Baseline GPU port, row-major DB                 | Global memory only          |
| SMEM      | Query vector cached in shared memory            | `extern __shared__`         |
| Coalesced | Database stored column-major (D×N)              | Stride-1 warp access        |
| Warp      | Fused top-K, no Q×N intermediate matrix         | `__shfl_down_sync`          |

### Memory Layout

The critical insight in Kernel 3: switching from row-major (N×D) to column-major (D×N) database storage.

- **Row-major** (Kernels 1, 2): thread `v` reads `db[v*D + dim]`. Consecutive threads are D floats apart → **uncoalesced**.
- **Column-major** (Kernels 3, 4): thread `v` reads `db[dim*N + v]`. Consecutive threads are 1 float apart → **coalesced** (single memory transaction per warp).

### Warp Reduction (Kernel 4)

```
K passes, each extracting one nearest neighbor:
  1. Each thread scans N/256 vectors, tracking local minimum
  2. __shfl_down_sync(0xFFFFFFFF, ...) reduces 32 lanes → warp minimum (no shared memory atomics)
  3. 8 warp leaders write candidates to shared memory
  4. Thread 0 selects global minimum, records it
  5. Next pass skips previously selected vectors
```

> **Note on cosine similarity:** Cosine distance reduces to L2 on unit-normalized vectors.
> Normalize your vectors with `vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)` before indexing.

> **Note on the GEMM approach:** An alternative formulation `||a-b||² = ||a||² + ||b||² - 2·a·b`
> enables batched matrix multiplication (used internally by FAISS). This was evaluated but not
> implemented — the direct kernel approach produces a cleaner per-optimization benchmark story.

## Project Structure

```
├── include/
│   ├── vector_index.h    # VectorIndex class interface
│   ├── kernels.h         # kernel launch signatures
│   └── cuda_utils.h      # CUDA_CHECK macro
├── src/
│   ├── vector_index.cu   # host orchestration, memory management, CPU reference
│   ├── kernel_naive.cu   # Kernel 1: global memory baseline
│   ├── kernel_smem.cu    # Kernel 2: shared memory query cache
│   ├── kernel_coalesced.cu # Kernel 3: column-major coalesced access
│   └── kernel_warp.cu    # Kernel 4: warp-level top-K reduction
├── python/
│   ├── bindings.cpp      # pybind11 wrapper
│   └── benchmark.py      # SIFT1M benchmark script
└── tests/
    ├── test_correctness.cpp  # all kernels vs CPU reference
    ├── test_perf.cu          # CUDA-event timing regression guard
    └── test_bindings.py      # Python API smoke tests
```

## Dependencies

- CUDA 12.x (Thrust bundled)
- CMake 3.18+
- Python 3.10+, pybind11, NumPy
