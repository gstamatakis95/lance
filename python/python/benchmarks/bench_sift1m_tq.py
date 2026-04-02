#!/usr/bin/env python3
"""SIFT1M benchmark for IVF_TQ vs IVF_PQ vs IVF_SQ.

Downloads SIFT1M (1M vectors, 128-dim) and measures:
  - Index build time (training + encoding)
  - Recall@10 at various nprobes
  - QPS (queries per second)

Run: python python/python/benchmarks/bench_sift1m_tq.py
"""

import struct
import tempfile
import time

import numpy as np
import pyarrow as pa

import lance


def read_fvecs(path: str) -> np.ndarray:
    """Read .fvecs format (int32 dim header + float32 data per row)."""
    with open(path, "rb") as f:
        data = f.read()
    # First 4 bytes of each vector is the dimension
    dim = struct.unpack("i", data[:4])[0]
    row_bytes = 4 + dim * 4  # 4 for dim header + dim*4 for floats
    n = len(data) // row_bytes
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        offset = i * row_bytes + 4  # skip dim header
        vectors[i] = np.frombuffer(data[offset : offset + dim * 4], dtype=np.float32)
    return vectors


def read_ivecs(path: str) -> np.ndarray:
    """Read .ivecs format (int32 dim header + int32 data per row)."""
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
    """Compute recall@k."""
    n = len(predicted_ids)
    recalls = []
    for i in range(n):
        gt = set(ground_truth_ids[i][:k])
        pred = set(predicted_ids[i][:k])
        recalls.append(len(gt & pred) / k)
    return np.mean(recalls)


def ensure_sift1m(base_dir: str = "/tmp/sift") -> str:
    """Download and extract SIFT1M if not already present. Returns path to data dir."""
    import os
    import tarfile
    import urllib.request

    data_dir = os.path.join(base_dir, "sift")
    if os.path.exists(os.path.join(data_dir, "sift_base.fvecs")):
        return data_dir

    os.makedirs(base_dir, exist_ok=True)
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    tar_path = os.path.join(base_dir, "sift.tar.gz")

    if not os.path.exists(tar_path):
        print(f"  Downloading SIFT1M from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"  Downloaded {os.path.getsize(tar_path) / 1e6:.0f} MB")

    print("  Extracting...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(base_dir)

    return data_dir


def main():
    sift_dir = ensure_sift1m()

    print("Loading SIFT1M dataset...")
    base = read_fvecs(f"{sift_dir}/sift_base.fvecs")
    queries = read_fvecs(f"{sift_dir}/sift_query.fvecs")
    gt = read_ivecs(f"{sift_dir}/sift_groundtruth.ivecs")
    print(f"  Base: {base.shape}, Queries: {queries.shape}, GT: {gt.shape}")

    # Write to Lance dataset
    uri = tempfile.mkdtemp(prefix="sift1m_lance_")
    print(f"  Writing Lance dataset to {uri}...")
    table = pa.table(
        {
            "id": pa.array(range(len(base)), type=pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(base.reshape(-1), type=pa.float32()), 128
            ),
        }
    )
    lance.write_dataset(table, uri, mode="overwrite")
    ds = lance.dataset(uri)
    print(f"  Dataset: {ds.count_rows()} rows")

    K = 10
    nprobes_list = [1, 4, 8, 16, 32, 64]
    num_partitions = 256

    configs = [
        ("IVF_PQ", {"num_sub_vectors": 16}),
        ("IVF_SQ", {}),
        ("IVF_TQ", {"num_bits": 4}),
        ("IVF_TQ", {"num_bits": 2}),
    ]

    print("\n" + "=" * 80)
    print("SIFT1M BENCHMARK (1M vectors, 128-dim, L2)")
    print("=" * 80)

    for index_type, extra_params in configs:
        label = index_type
        if "num_bits" in extra_params:
            label += f"_{extra_params['num_bits']}bit"
        if "num_sub_vectors" in extra_params:
            label += f"_M{extra_params['num_sub_vectors']}"

        print(f"\n--- {label} (partitions={num_partitions}) ---")

        # Build index
        t0 = time.time()
        ds.create_index(
            "vector",
            index_type=index_type,
            name=label.lower(),
            num_partitions=num_partitions,
            replace=True,
            **extra_params,
        )
        build_time = time.time() - t0
        print(f"  Build time: {build_time:.2f}s")

        # Reload dataset to pick up new index
        ds = lance.dataset(uri)

        # Search at various nprobes
        print(f"  {'nprobes':>8} {'recall@10':>10} {'QPS':>10} {'latency_ms':>12}")
        for nprobes in nprobes_list:
            predicted_ids = []
            t0 = time.time()
            for q in queries[:1000]:  # use first 1000 queries
                results = ds.to_table(
                    nearest={
                        "column": "vector",
                        "q": q,
                        "k": K,
                        "nprobes": nprobes,
                    },
                    columns=["id"],
                )
                predicted_ids.append(results.column("id").to_pylist())
            elapsed = time.time() - t0
            n_queries = len(predicted_ids)

            recall = compute_recall(predicted_ids, gt, K)
            qps = n_queries / elapsed
            latency_ms = elapsed / n_queries * 1000

            print(f"  {nprobes:>8} {recall:>10.4f} {qps:>10.1f} {latency_ms:>12.2f}")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
