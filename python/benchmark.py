"""
SIFT1M benchmark for all 4 CUDA kernels.

Usage:
    python python/benchmark.py --data-dir /path/to/sift

Download SIFT1M from:
    http://corpus-texmex.irisa.fr/  (sift.tar.gz, ~500MB)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import _vsearch as vs


# ── Dataset loaders ────────────────────────────────────────────────────────────
def load_fvecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    dim  = data[0]
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].view(np.float32).copy()


def load_ivecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    dim  = data[0]
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].copy()


# ── Timing helper ──────────────────────────────────────────────────────────────
def time_search(index, queries, k, kernel, warmup=1, runs=3) -> float:
    for _ in range(warmup):
        index.search(queries, k, kernel=kernel)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        index.search(queries, k, kernel=kernel)
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)  # best-of-N to reduce noise


# ── Recall computation ─────────────────────────────────────────────────────────
def compute_recall(got_idx: np.ndarray, gt_idx: np.ndarray, k: int) -> float:
    """Recall@k: fraction of queries where the ground-truth NN appears in top-k results."""
    n_correct = sum(
        1 for qi in range(len(got_idx))
        if gt_idx[qi, 0] in got_idx[qi, :k]
    )
    return n_correct / len(got_idx)


# ── CPU baseline ───────────────────────────────────────────────────────────────
def cpu_search(db: np.ndarray, queries: np.ndarray, k: int):
    """Numpy brute-force L2 search. Returns (ms, indices)."""
    t0 = time.perf_counter()
    all_idx = []
    for qi in range(len(queries)):
        diffs = db - queries[qi]          # (N, D)
        dists = (diffs ** 2).sum(axis=1)  # (N,)
        part  = np.argpartition(dists, k)[:k]
        part  = part[np.argsort(dists[part])]
        all_idx.append(part)
    ms = (time.perf_counter() - t0) * 1000
    return ms, np.stack(all_idx)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing sift_base.fvecs, etc.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=100,
                        help="Number of query vectors to benchmark")
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    print(f"Loading SIFT1M from {data_dir} ...")

    db      = load_fvecs(str(data_dir / "sift_base.fvecs"))
    queries = load_fvecs(str(data_dir / "sift_query.fvecs"))[:args.n_queries]
    gt      = load_ivecs(str(data_dir / "sift_groundtruth.ivecs"))[:args.n_queries]

    print(f"  DB: {db.shape}  Queries: {queries.shape}  GT: {gt.shape}")

    # ── Build index ────────────────────────────────────────────────────────────
    print("Building index ...")
    index = vs.VectorIndex(db.shape[1])
    index.add(db)

    # ── CPU baseline ───────────────────────────────────────────────────────────
    print("Running CPU baseline (this may take a minute) ...")
    cpu_ms, cpu_idx = cpu_search(db, queries, args.k)
    cpu_recall = compute_recall(cpu_idx, gt, args.k)

    # ── GPU kernels ────────────────────────────────────────────────────────────
    kernels = ["naive", "smem", "coalesced", "warp"]
    results = []

    for kernel in kernels:
        ms = time_search(index, queries, args.k, kernel)
        _, idx = index.search(queries, args.k, kernel=kernel)
        recall = compute_recall(idx, gt, args.k)
        speedup = cpu_ms / ms
        results.append({
            "kernel": kernel, "time_ms": round(ms, 2),
            "speedup": round(speedup, 1), "recall": round(recall * 100, 1),
        })

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\nBenchmark Results  (N={len(db):,}, D={db.shape[1]}, "
          f"Q={len(queries)}, K={args.k})\n")
    print(f"{'Version':<20} {'Time(ms)':>10} {'Speedup':>10} {'Recall@%d' % args.k:>12}")
    print("-" * 55)
    print(f"{'CPU baseline':<20} {cpu_ms:>10.1f} {'1x':>10} {cpu_recall*100:>11.1f}%")
    for r in results:
        print(f"{'CUDA ' + r['kernel']:<20} {r['time_ms']:>10.1f} "
              f"{r['speedup']:>9.1f}x {r['recall']:>11.1f}%")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "config": {"n": len(db), "d": int(db.shape[1]),
                   "q": len(queries), "k": args.k},
        "cpu_baseline": {"time_ms": round(cpu_ms, 2), "recall": round(cpu_recall * 100, 1)},
        "kernels": results,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
