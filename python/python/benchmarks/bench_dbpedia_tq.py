#!/usr/bin/env python3
"""DBpedia-OpenAI benchmark (1536-dim) for IVF_TQ vs IVF_PQ.

This is the same dataset used in the TurboQuant paper (Section 4.4, Figure 5).
At dim=1536, PQ's subvector independence assumption degrades significantly,
while TurboQuant's rotation decorrelates all dimensions optimally.

Downloads 100K vectors from Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M.

Run: python python/python/benchmarks/bench_dbpedia_tq.py
"""

import os
import tempfile
import time

import numpy as np
import pyarrow as pa

import lance

N_BASE = 100_000
N_QUERIES = 1_000
DIM = 1536
K = 10


def load_dbpedia(n_base: int, n_queries: int, cache_dir: str = "/tmp/dbpedia_cache"):
    """Load DBpedia-OpenAI vectors, caching to disk."""
    base_path = os.path.join(cache_dir, f"base_{n_base}.npy")
    query_path = os.path.join(cache_dir, f"queries_{n_queries}.npy")

    if os.path.exists(base_path) and os.path.exists(query_path):
        print("  Loading from cache...")
        return np.load(base_path), np.load(query_path)

    print(f"  Downloading {n_base + n_queries} vectors from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        split="train",
        streaming=True,
    )

    total = n_base + n_queries
    vectors = np.zeros((total, DIM), dtype=np.float32)
    for i, row in enumerate(ds):
        if i >= total:
            break
        vectors[i] = row["text-embedding-3-large-1536-embedding"]
        if (i + 1) % 10000 == 0:
            print(f"    Loaded {i + 1}/{total}...")

    base = vectors[:n_base]
    queries = vectors[n_base : n_base + n_queries]

    os.makedirs(cache_dir, exist_ok=True)
    np.save(base_path, base)
    np.save(query_path, queries)
    print(f"  Cached to {cache_dir}")

    return base, queries


def brute_force_knn(base, queries, k):
    """Exact KNN ground truth via batched computation."""
    print(f"  Computing ground truth ({len(queries)} queries)...")
    gt = np.zeros((len(queries), k), dtype=np.int64)
    # Batch to avoid memory issues
    batch_size = 100
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        q_batch = queries[start:end]
        # ||q - x||^2 = ||q||^2 + ||x||^2 - 2*q@x^T
        q_norms = np.sum(q_batch**2, axis=1, keepdims=True)
        x_norms = np.sum(base**2, axis=1, keepdims=True).T
        dists = q_norms + x_norms - 2 * q_batch @ base.T
        gt[start:end] = np.argsort(dists, axis=1)[:, :k]
    return gt


def compute_recall(predicted_ids, ground_truth_ids, k):
    n = len(predicted_ids)
    recalls = []
    for i in range(n):
        gt = set(ground_truth_ids[i][:k])
        pred = set(predicted_ids[i][:k])
        recalls.append(len(gt & pred) / k)
    return np.mean(recalls)


def search(ds, queries, k, nprobes, n_queries=None):
    if n_queries is None:
        n_queries = len(queries)
    predicted_ids = []
    t0 = time.time()
    for q in queries[:n_queries]:
        results = ds.to_table(
            nearest={"column": "vector", "q": q, "k": k, "nprobes": nprobes},
            columns=["id"],
        )
        predicted_ids.append(results.column("id").to_pylist())
    elapsed = time.time() - t0
    return predicted_ids, elapsed


def main():
    print("=" * 80)
    print(f"DBpedia-OpenAI BENCHMARK ({N_BASE // 1000}K vectors, dim={DIM}, L2)")
    print("Paper reference: Zandieh et al., ICLR 2026, Section 4.4, Figure 5")
    print("=" * 80)

    base, queries = load_dbpedia(N_BASE, N_QUERIES)
    print(f"  Base: {base.shape}, Queries: {queries.shape}")

    gt = brute_force_knn(base, queries, K)

    uri = tempfile.mkdtemp(prefix="dbpedia_lance_")
    print(f"  Writing Lance dataset to {uri}...")
    table = pa.table(
        {
            "id": pa.array(range(N_BASE), type=pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(base.reshape(-1), type=pa.float32()), DIM
            ),
        }
    )
    lance.write_dataset(table, uri, mode="overwrite")

    num_partitions = 256
    nprobes_list = [1, 4, 8, 16, 32, 64]
    n_search_queries = 500  # use 500 queries for speed

    configs = [
        ("IVF_PQ", {"num_sub_vectors": 96}, "IVF_PQ M=96 (96B/vec)"),
        ("IVF_PQ", {"num_sub_vectors": 48}, "IVF_PQ M=48 (48B/vec)"),
        ("IVF_PQ", {"num_sub_vectors": 16}, "IVF_PQ M=16 (16B/vec)"),
        ("IVF_SQ", {}, "IVF_SQ 8bit (1536B/vec)"),
        ("IVF_TQ", {"num_bits": 4}, "IVF_TQ 4bit (772B/vec)"),
        ("IVF_TQ", {"num_bits": 2}, "IVF_TQ 2bit (388B/vec)"),
        ("IVF_TQ", {"num_bits": 1}, "IVF_TQ 1bit (196B/vec)"),
    ]

    all_results = {}

    for index_type, extra_params, label in configs:
        print(f"\n--- {label} ---")

        ds = lance.dataset(uri)
        t0 = time.time()
        ds.create_index(
            "vector",
            index_type=index_type,
            name=label.split()[0].lower() + "_" + label.split()[1].lower(),
            num_partitions=num_partitions,
            replace=True,
            **extra_params,
        )
        build_time = time.time() - t0
        print(f"  Build: {build_time:.2f}s")

        ds = lance.dataset(uri)

        recall_results = []
        for nprobes in nprobes_list:
            predicted_ids, elapsed = search(ds, queries, K, nprobes, n_search_queries)
            recall = compute_recall(predicted_ids, gt[:n_search_queries], K)
            qps = n_search_queries / elapsed
            recall_results.append((nprobes, recall, qps))
            print(f"  nprobes={nprobes:>3}: recall@{K}={recall:.4f}  QPS={qps:.0f}")

        all_results[label] = {"build_time": build_time, "recall": recall_results}

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: recall@10 at nprobes=32")
    print("=" * 80)
    print(f"{'Method':<30} {'Build(s)':>8} {'Recall@10':>10} {'QPS':>8} {'Bytes/vec':>10}")
    print("-" * 80)
    for label, data in all_results.items():
        build = data["build_time"]
        # Find nprobes=32 result
        for np_, recall, qps in data["recall"]:
            if np_ == 32:
                bytes_str = label.split("(")[1].rstrip(")") if "(" in label else "?"
                print(f"{label:<30} {build:>8.2f} {recall:>10.4f} {qps:>8.0f} {bytes_str:>10}")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
