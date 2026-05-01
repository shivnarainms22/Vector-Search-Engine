import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pytest
import _vsearch as vs

D = 128
N = 500
Q = 5
K = 10

@pytest.fixture
def index():
    rng = np.random.default_rng(42)
    db = rng.standard_normal((N, D)).astype(np.float32)
    idx = vs.VectorIndex(D)
    idx.add(db)
    return idx

def test_search_output_shapes(index):
    rng = np.random.default_rng(99)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    for kernel in ("naive", "smem", "coalesced", "warp"):
        dist, idx = index.search(queries, K, kernel=kernel)
        assert dist.shape == (Q, K), f"{kernel}: dist shape {dist.shape}"
        assert idx.shape  == (Q, K), f"{kernel}: idx shape {idx.shape}"
        assert dist.dtype == np.float32, f"{kernel}: dist dtype {dist.dtype}"
        assert idx.dtype  == np.int32,   f"{kernel}: idx dtype {idx.dtype}"

def test_distances_non_negative(index):
    rng = np.random.default_rng(7)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    dist, _ = index.search(queries, K, kernel="warp")
    assert (dist >= 0).all(), "Distances must be non-negative"

def test_indices_in_range(index):
    rng = np.random.default_rng(8)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    _, idx = index.search(queries, K, kernel="warp")
    assert ((idx >= 0) & (idx < N)).all(), "Indices out of range"

def test_k_gt_n_raises():
    rng = np.random.default_rng(0)
    db = rng.standard_normal((10, D)).astype(np.float32)
    idx = vs.VectorIndex(D)
    idx.add(db)
    queries = rng.standard_normal((1, D)).astype(np.float32)
    with pytest.raises(Exception):
        idx.search(queries, 20, kernel="warp")

def test_wrong_dim_raises(index):
    bad = np.ones((Q, D + 1), dtype=np.float32)
    with pytest.raises(Exception):
        index.search(bad, K, kernel="warp")

def test_kernel_results_consistent(index):
    rng = np.random.default_rng(55)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    dist_ref, idx_ref = index.search(queries, K, kernel="naive")
    for kernel in ("smem", "coalesced", "warp"):
        dist, idx = index.search(queries, K, kernel=kernel)
        np.testing.assert_allclose(dist, dist_ref, rtol=1e-3,
                                   err_msg=f"{kernel} distances differ from naive")
