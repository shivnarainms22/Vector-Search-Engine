# Vector Search Engine

## Tech Stack
- CUDA 12.x, C++17
- Python 3.10+, pybind11
- CMake 3.18+, Thrust (bundled with CUDA)

## Build
```bash
pip install pybind11
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;86"
cmake --build build --parallel
```

## Run correctness tests
```bash
./build/test_correctness
```

## Run performance tests
```bash
./build/test_perf
```

## Run Python tests
```bash
pip install pytest numpy
python -m pytest tests/test_bindings.py -v
```

## Run benchmark (requires SIFT1M dataset)
```bash
python python/benchmark.py --data-dir /path/to/sift
```
