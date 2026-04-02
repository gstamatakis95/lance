# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""TurboQuant (IVF_TQ) benchmarks.

Reproduces key results from Zandieh et al. (ICLR 2026):
  - Build time comparison: TQ vs PQ training time (paper Table 2)
  - Recall comparison: IVF_TQ vs IVF_PQ vs IVF_SQ (paper Fig. 5)

Run: pytest python/python/benchmarks/test_tq_index.py -v --benchmark-disable
  (or with --benchmark-enable for full criterion-style benchmarks)
"""

import tempfile
import time

import numpy as np
import pyarrow as pa
import pytest

import lance


def generate_random_dataset(n: int, dim: int, seed: int = 42) -> str:
    """Generate a random dataset with unit vectors and return its URI."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    table = pa.table(
        {
            "id": pa.array(range(n), type=pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), type=pa.float32()), dim
            ),
        }
    )
    uri = tempfile.mkdtemp()
    lance.write_dataset(table, uri, mode="overwrite")
    return uri


def compute_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int):
    """Brute-force exact KNN for ground truth."""
    # L2 distances
    dists = np.sum((vectors[np.newaxis, :, :] - queries[:, np.newaxis, :]) ** 2, axis=2)
    return np.argsort(dists, axis=1)[:, :k]


def compute_recall(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute recall@k."""
    n_queries = predicted.shape[0]
    recalls = []
    for i in range(n_queries):
        gt_set = set(ground_truth[i])
        pred_set = set(predicted[i])
        recalls.append(len(gt_set & pred_set) / len(gt_set))
    return np.mean(recalls)


N_VECTORS = 10_000
DIM = 128
N_QUERIES = 100
K = 10


@pytest.fixture(scope="module")
def test_dataset():
    """Create a test dataset for benchmarks."""
    uri = generate_random_dataset(N_VECTORS, DIM)
    ds = lance.dataset(uri)
    return ds, uri


@pytest.fixture(scope="module")
def ground_truth(test_dataset):
    """Compute ground truth for recall measurement."""
    ds, _ = test_dataset
    vectors = (
        ds.to_table(columns=["vector"])
        .column("vector")
        .to_numpy(zero_copy_only=False)
    )
    # Stack into (n, dim)
    vectors = np.stack([np.array(v) for v in vectors])

    rng = np.random.default_rng(99)
    query_indices = rng.choice(N_VECTORS, N_QUERIES, replace=False)
    queries = vectors[query_indices]
    gt = compute_ground_truth(vectors, queries, K)
    return queries, gt, query_indices


class TestBuildTime:
    """Compare index build times (reproduces paper Table 2)."""

    def test_tq_build_time(self, test_dataset):
        """TurboQuant build should be near-instant (no training needed)."""
        ds, uri = test_dataset

        t0 = time.time()
        ds.create_index(
            "vector",
            index_type="IVF_TQ",
            name="tq_bench",
            num_partitions=32,
            num_bits=4,
            replace=True,
        )
        tq_time = time.time() - t0
        print(f"\nIVF_TQ build: {tq_time:.2f}s ({N_VECTORS} vectors, dim={DIM})")

    def test_pq_build_time(self, test_dataset):
        """PQ build includes expensive codebook training."""
        ds, uri = test_dataset

        t0 = time.time()
        ds.create_index(
            "vector",
            index_type="IVF_PQ",
            name="pq_bench",
            num_partitions=32,
            num_sub_vectors=16,
            replace=True,
        )
        pq_time = time.time() - t0
        print(f"\nIVF_PQ build: {pq_time:.2f}s ({N_VECTORS} vectors, dim={DIM})")

    def test_sq_build_time(self, test_dataset):
        """SQ build is relatively fast."""
        ds, uri = test_dataset

        t0 = time.time()
        ds.create_index(
            "vector",
            index_type="IVF_SQ",
            name="sq_bench",
            num_partitions=32,
            replace=True,
        )
        sq_time = time.time() - t0
        print(f"\nIVF_SQ build: {sq_time:.2f}s ({N_VECTORS} vectors, dim={DIM})")


class TestRecall:
    """Compare recall@10 across index types (reproduces paper Fig. 5)."""

    def _search_and_recall(self, ds, index_name, queries, ground_truth_ids):
        """Search the index and compute recall."""
        predicted = []
        for q in queries:
            results = ds.to_table(
                nearest={"column": "vector", "q": q, "k": K, "nprobes": 8},
            )
            predicted.append(results.column("id").to_pylist())

        predicted = np.array(predicted)
        recall = compute_recall(predicted, ground_truth_ids)
        return recall

    def test_tq_recall(self, test_dataset, ground_truth):
        """IVF_TQ should achieve competitive recall."""
        ds, uri = test_dataset
        queries, gt, _ = ground_truth

        ds.create_index(
            "vector",
            index_type="IVF_TQ",
            name="tq_recall",
            num_partitions=32,
            num_bits=4,
            replace=True,
        )

        ds_reloaded = lance.dataset(uri)
        recall = self._search_and_recall(ds_reloaded, "tq_recall", queries, gt)
        print(f"\nIVF_TQ recall@{K}: {recall:.3f}")
        assert recall >= 0.5, f"TQ recall {recall:.3f} below minimum threshold 0.5"

    def test_pq_recall(self, test_dataset, ground_truth):
        """IVF_PQ recall for comparison."""
        ds, uri = test_dataset
        queries, gt, _ = ground_truth

        ds.create_index(
            "vector",
            index_type="IVF_PQ",
            name="pq_recall",
            num_partitions=32,
            num_sub_vectors=16,
            replace=True,
        )

        ds_reloaded = lance.dataset(uri)
        recall = self._search_and_recall(ds_reloaded, "pq_recall", queries, gt)
        print(f"\nIVF_PQ recall@{K}: {recall:.3f}")
