#!/usr/bin/env python3
"""Quantizer quality benchmark — isolates quantizer differences from IVF effects.

Three experiments:
  1. SIFT1M with aggressive IVF (1024 partitions, low nprobes)
  2. Synthetic 768-dim (where PQ subvector independence hurts)
  3. Direct distance estimation quality (no IVF, pure quantizer comparison)

Run: python python/python/benchmarks/bench_quantizer_quality.py
"""

import struct
import tempfile
import time
import os

import numpy as np
import pyarrow as pa

import lance


def read_fvecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    dim = struct.unpack("i", data[:4])[0]
    row_bytes = 4 + dim * 4
    n = len(data) // row_bytes
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        offset = i * row_bytes + 4
        vectors[i] = np.frombuffer(data[offset : offset + dim * 4], dtype=np.float32)
    return vectors


def read_ivecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    dim = struct.unpack("i", data[:4])[0]
    row_bytes = 4 + dim * 4
    n = len(data) // row_bytes
    vectors = np.zeros((n, dim), dtype=np.int32)
    for i in range(n):
        offset = i * row_bytes + 4
        vectors[i] = np.frombuffer(data[offset : offset + dim * 4], dtype=np.int32)
    return vectors


def compute_recall(predicted_ids, ground_truth_ids, k):
    n = len(predicted_ids)
    recalls = []
    for i in range(n):
        gt = set(ground_truth_ids[i][:k])
        pred = set(predicted_ids[i][:k])
        recalls.append(len(gt & pred) / k)
    return np.mean(recalls)


def make_lance_dataset(vectors, uri):
    n, dim = vectors.shape
    table = pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1), type=pa.float32()), dim
        ),
    })
    lance.write_dataset(table, uri, mode="overwrite")
    return lance.dataset(uri)


def brute_force_knn(base, queries, k):
    """Compute exact KNN ground truth."""
    gt = np.zeros((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        dists = np.sum((base - q) ** 2, axis=1)
        gt[i] = np.argsort(dists)[:k]
    return gt


def run_recall_benchmark(ds, uri, queries, gt, configs, num_partitions, nprobes_list, k=10):
    """Run recall benchmark for multiple configs."""
    n_queries = min(500, len(queries))
    results = {}

    for index_type, extra_params, label in configs:
        print(f"\n  {label}:")

        # Build
        t0 = time.time()
        ds.create_index(
            "vector",
            index_type=index_type,
            name=label.lower().replace(" ", "_"),
            num_partitions=num_partitions,
            replace=True,
            **extra_params,
        )
        build_time = time.time() - t0
        print(f"    Build: {build_time:.2f}s")

        ds_fresh = lance.dataset(uri)

        row_results = []
        for nprobes in nprobes_list:
            predicted_ids = []
            for q in queries[:n_queries]:
                r = ds_fresh.to_table(
                    nearest={"column": "vector", "q": q, "k": k, "nprobes": nprobes},
                    columns=["id"],
                )
                predicted_ids.append(r.column("id").to_pylist())

            recall = compute_recall(predicted_ids, gt, k)
            row_results.append((nprobes, recall))
            print(f"    nprobes={nprobes:>3}: recall@{k} = {recall:.4f}")

        results[label] = {"build_time": build_time, "recall": row_results}

    return results


# ============================================================================
# EXPERIMENT 1: SIFT1M with aggressive IVF (1024 partitions)
# ============================================================================
def experiment_sift1m_aggressive():
    sift_dir = "/tmp/sift/sift"
    if not os.path.exists(f"{sift_dir}/sift_base.fvecs"):
        print("  SIFT1M not found at /tmp/sift/sift — skipping")
        print("  Download: curl -sL ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz | tar xz -C /tmp/sift")
        return

    print("Loading SIFT1M...")
    base = read_fvecs(f"{sift_dir}/sift_base.fvecs")
    queries = read_fvecs(f"{sift_dir}/sift_query.fvecs")
    gt = read_ivecs(f"{sift_dir}/sift_groundtruth.ivecs")

    uri = tempfile.mkdtemp(prefix="sift1m_agg_")
    ds = make_lance_dataset(base, uri)

    configs = [
        ("IVF_PQ", {"num_sub_vectors": 16}, "IVF_PQ M=16"),
        ("IVF_PQ", {"num_sub_vectors": 8}, "IVF_PQ M=8"),
        ("IVF_SQ", {}, "IVF_SQ 8bit"),
        ("IVF_TQ", {"num_bits": 4}, "IVF_TQ 4bit"),
        ("IVF_TQ", {"num_bits": 2}, "IVF_TQ 2bit"),
        ("IVF_TQ", {"num_bits": 1}, "IVF_TQ 1bit"),
    ]

    # 1024 partitions = ~1000 vectors per partition
    # Low nprobes forces quantizer to rank accurately within small candidate sets
    run_recall_benchmark(ds, uri, queries, gt, configs,
                         num_partitions=1024, nprobes_list=[1, 2, 4, 8, 16, 32])


# ============================================================================
# EXPERIMENT 2: Synthetic 768-dim (where PQ subvector independence hurts)
# ============================================================================
def experiment_high_dim():
    dim = 768
    n_base = 100_000
    n_queries = 500
    k = 10

    print(f"Generating synthetic dataset: {n_base} vectors, dim={dim}...")
    rng = np.random.default_rng(42)

    # Create correlated vectors (not uniform random — PQ should struggle)
    # Use a low-rank structure: vectors = A @ z where A is 768x32
    rank = 32
    A = rng.standard_normal((dim, rank)).astype(np.float32)
    z_base = rng.standard_normal((n_base, rank)).astype(np.float32)
    base = (z_base @ A.T).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    base = base / norms

    z_queries = rng.standard_normal((n_queries, rank)).astype(np.float32)
    queries = (z_queries @ A.T).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / q_norms

    print("Computing ground truth...")
    gt = brute_force_knn(base, queries, k)

    uri = tempfile.mkdtemp(prefix="highdim_")
    ds = make_lance_dataset(base, uri)

    configs = [
        ("IVF_PQ", {"num_sub_vectors": 48}, "IVF_PQ M=48"),
        ("IVF_PQ", {"num_sub_vectors": 16}, "IVF_PQ M=16"),
        ("IVF_SQ", {}, "IVF_SQ 8bit"),
        ("IVF_TQ", {"num_bits": 4}, "IVF_TQ 4bit"),
        ("IVF_TQ", {"num_bits": 2}, "IVF_TQ 2bit"),
    ]

    run_recall_benchmark(ds, uri, queries, gt, configs,
                         num_partitions=256, nprobes_list=[1, 4, 8, 16, 32])


# ============================================================================
# EXPERIMENT 3: Storage size comparison
# ============================================================================
def experiment_storage_sizes():
    dim = 768
    configs = [
        ("IVF_PQ M=48", 48),            # 48 bytes/vec
        ("IVF_PQ M=16", 16),            # 16 bytes/vec
        ("IVF_SQ 8bit", dim),           # 768 bytes/vec
        ("IVF_TQ 4bit", dim * 4 // 8 + 4),  # 388 bytes/vec
        ("IVF_TQ 2bit", dim * 2 // 8 + 4),  # 196 bytes/vec
        ("IVF_TQ 1bit", dim * 1 // 8 + 4),  # 100 bytes/vec
        ("fp32 (uncompressed)", dim * 4),     # 3072 bytes/vec
    ]

    print(f"\n  {'Method':<25} {'Bytes/vec':>10} {'1M vectors':>12} {'Compression':>12}")
    print("  " + "-" * 62)
    fp32_size = dim * 4
    for name, bytes_per_vec in configs:
        total_mb = bytes_per_vec * 1_000_000 / (1024 * 1024)
        compression = f"{fp32_size / bytes_per_vec:.1f}x"
        print(f"  {name:<25} {bytes_per_vec:>10} {total_mb:>10.0f} MB {compression:>12}")


def main():
    print("=" * 80)
    print("QUANTIZER QUALITY BENCHMARK")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: SIFT1M with 1024 partitions (isolate quantizer quality)")
    print("=" * 80)
    experiment_sift1m_aggressive()

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Synthetic 768-dim correlated vectors")
    print("=" * 80)
    experiment_high_dim()

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Storage size comparison (d=768)")
    print("=" * 80)
    experiment_storage_sizes()

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
