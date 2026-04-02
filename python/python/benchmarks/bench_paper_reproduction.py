#!/usr/bin/env python3
"""Reproduce TurboQuant paper experimental results (Zandieh et al., ICLR 2026).

Reproduces:
  - Section 4.1: Distortion validation (MSE + inner product bias/variance)
  - Section 4.4, Figure 5: Recall@1@k using quantized distances (no IVF)

The paper measures recall by quantizing ALL database vectors, then computing
quantized distances to rank candidates. NO IVF partitioning is used for the
recall measurement — it's a pure quantizer quality test.

Datasets:
  - DBpedia-OpenAI d=1536 (100K base, 1K queries)
  - GloVe d=200 (if available)

Run: python python/python/benchmarks/bench_paper_reproduction.py
"""

import os
import time

import numpy as np


def load_dbpedia(n_base=100_000, n_queries=1_000, cache_dir="/tmp/dbpedia_cache"):
    """Load DBpedia-OpenAI 1536-dim vectors."""
    base_path = os.path.join(cache_dir, f"base_{n_base}.npy")
    query_path = os.path.join(cache_dir, f"queries_{n_queries}.npy")

    if os.path.exists(base_path) and os.path.exists(query_path):
        return np.load(base_path), np.load(query_path)

    print("  Downloading from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        split="train",
        streaming=True,
    )
    total = n_base + n_queries
    dim = 1536
    vectors = np.zeros((total, dim), dtype=np.float32)
    for i, row in enumerate(ds):
        if i >= total:
            break
        vectors[i] = row["text-embedding-3-large-1536-embedding"]
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{total}...")

    base, queries = vectors[:n_base], vectors[n_base:]
    os.makedirs(cache_dir, exist_ok=True)
    np.save(base_path, base)
    np.save(query_path, queries)
    return base, queries


def brute_force_knn_ip(base, queries, k):
    """Exact KNN using inner product (higher = more similar)."""
    gt = np.zeros((len(queries), k), dtype=np.int64)
    batch_size = 100
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        sims = queries[start:end] @ base.T
        gt[start:end] = np.argsort(-sims, axis=1)[:, :k]  # descending
    return gt


def recall_at_1_at_k(predicted_ranks, gt, k_values):
    """Compute recall@1@k: is the true NN in the top-k approximate results?"""
    results = {}
    for k in k_values:
        correct = 0
        for i in range(len(gt)):
            true_nn = gt[i, 0]  # true nearest neighbor
            if true_nn in predicted_ranks[i][:k]:
                correct += 1
        results[k] = correct / len(gt)
    return results


# ============================================================================
# TurboQuant quantizer (using the Rust implementation via lance)
# ============================================================================
def tq_quantize_and_rank(base, queries, num_bits, seed=42):
    """Quantize base with TQ via Lance Rust backend, rank by reconstructed inner product."""
    import pyarrow as pa
    from lance.lance import indices as lance_indices

    n, dim = base.shape

    # Train TQ model (via Rust — near-instant)
    tq_model = lance_indices.train_tq_model(dim, num_bits, seed)
    rotation = np.array(tq_model.rotation_matrix.to_pylist(), dtype=np.float32)

    # Compute codebook via numerical integration (same as Rust codebook.rs)
    # For high dims, Beta((d-1)/2, (d-1)/2) ≈ N(0, 1/d)
    # Use scipy for the Lloyd-Max codebook
    from scipy import stats, integrate

    alpha = (dim - 1) / 2
    k = 2**num_bits

    # Quantile initialization
    qpts = np.linspace(0.5 / k, 1 - 0.5 / k, k)
    centroids = 2 * stats.beta.ppf(qpts, alpha, alpha) - 1

    def beta_pdf(x):
        return stats.beta.pdf((x + 1) / 2, alpha, alpha) / 2

    def cond_mean(a, b):
        num, _ = integrate.quad(lambda x: x * beta_pdf(x), a, b)
        den, _ = integrate.quad(beta_pdf, a, b)
        return num / den if den > 1e-15 else (a + b) / 2

    for _ in range(200):
        boundaries = np.empty(k + 1)
        boundaries[0], boundaries[-1] = -1.0, 1.0
        boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2
        new_c = np.array([cond_mean(boundaries[i], boundaries[i + 1]) for i in range(k)])
        if np.max(np.abs(new_c - centroids)) < 1e-10:
            break
        centroids = new_c

    boundaries = np.empty(k + 1)
    boundaries[0], boundaries[-1] = -1.0, 1.0
    boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2
    centroids = centroids.astype(np.float32)

    # Quantize: normalize, rotate, scalar quantize, dequantize, inverse rotate
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-10, 1.0, norms)
    base_hat = base / safe_norms  # unit sphere
    rotated = base_hat @ rotation.T  # y = x @ R^T (same as Π·x in paper)
    indices = np.searchsorted(boundaries[1:-1], rotated).astype(np.uint8)
    y_hat = centroids[indices]  # dequantize in rotated space
    reconstructed = (y_hat @ rotation) * norms  # inverse rotate + rescale

    # Rank by inner product
    n_queries = len(queries)
    rankings = np.zeros((n_queries, len(base)), dtype=np.int64)
    batch_size = 50
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        sims = queries[start:end] @ reconstructed.T
        rankings[start:end] = np.argsort(-sims, axis=1)

    return rankings


def pq_quantize_and_rank(base, queries, num_subvectors, num_bits_pq=8):
    """Quantize base vectors with PQ, compute approximate distances, rank."""
    from sklearn.cluster import KMeans

    dim = base.shape[1]
    sub_dim = dim // num_subvectors

    # Train PQ codebooks
    codebooks = []
    for m in range(num_subvectors):
        sub_vectors = base[:, m * sub_dim : (m + 1) * sub_dim]
        sample = sub_vectors[np.random.choice(len(sub_vectors), min(10000, len(sub_vectors)), replace=False)]
        kmeans = KMeans(n_clusters=2**num_bits_pq, max_iter=50, n_init=1, random_state=42)
        kmeans.fit(sample)
        codebooks.append(kmeans.cluster_centers_)

    # Encode base vectors
    codes = np.zeros((len(base), num_subvectors), dtype=np.uint8)
    for m in range(num_subvectors):
        sub_vectors = base[:, m * sub_dim : (m + 1) * sub_dim]
        dists = np.sum((sub_vectors[:, np.newaxis] - codebooks[m][np.newaxis]) ** 2, axis=2)
        codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

    # Reconstruct
    reconstructed = np.zeros_like(base)
    for m in range(num_subvectors):
        reconstructed[:, m * sub_dim : (m + 1) * sub_dim] = codebooks[m][codes[:, m]]

    # Rank
    n_queries = len(queries)
    rankings = np.zeros((n_queries, len(base)), dtype=np.int64)
    batch_size = 50
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        sims = queries[start:end] @ reconstructed.T
        rankings[start:end] = np.argsort(-sims, axis=1)

    return rankings


def naive_quantize_and_rank(base, queries, num_bits):
    """Uniform scalar quantization (baseline), then rank."""
    # Per-dimension min/max
    mins = base.min(axis=0)
    maxs = base.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    levels = 2**num_bits
    codes = np.clip(((base - mins) / ranges * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)
    reconstructed = codes.astype(np.float32) / (levels - 1) * ranges + mins

    n_queries = len(queries)
    rankings = np.zeros((n_queries, len(base)), dtype=np.int64)
    batch_size = 50
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        sims = queries[start:end] @ reconstructed.T
        rankings[start:end] = np.argsort(-sims, axis=1)

    return rankings


def main():
    print("=" * 80)
    print("TURBOQUANT PAPER REPRODUCTION")
    print("Zandieh et al., ICLR 2026, arXiv:2504.19874")
    print("=" * 80)

    # ================================================================
    # Section 4.1: Distortion validation
    # ================================================================
    print("\n" + "=" * 80)
    print("Section 4.1: MSE Distortion Validation (Table in paper)")
    print("=" * 80)

    from lance.lance import indices as lance_indices
    from scipy import stats, integrate

    # Paper Theorem 1 values (Section 3.1, proven upper bounds)
    paper_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    # Paper Theorem 3 lower bounds (information-theoretic, no algorithm can do better)
    lower_bounds = {1: 0.25, 2: 0.0625, 3: 0.015625, 4: 0.00390625}
    n_samples = 2000

    print(f"\n{'dim':>6} {'bits':>5} {'MSE_ours':>10} {'paper_MSE':>10} {'lower_bnd':>10} {'ours/paper':>10} {'ours/lower':>11}")
    print("-" * 75)

    for dim in [128, 1536]:
        for b in [1, 2, 3, 4]:
            # Use Lance Rust backend for rotation matrix
            tq = lance_indices.train_tq_model(dim, b, 42)
            rotation = np.array(tq.rotation_matrix.to_pylist(), dtype=np.float32)

            # Codebook via scipy (same algorithm as Rust codebook.rs)
            alpha = (dim - 1) / 2
            k = 2**b
            qpts = np.linspace(0.5 / k, 1 - 0.5 / k, k)
            centroids = 2 * stats.beta.ppf(qpts, alpha, alpha) - 1

            def beta_pdf(x):
                return stats.beta.pdf((x + 1) / 2, alpha, alpha) / 2

            def cond_mean(a, b_):
                num, _ = integrate.quad(lambda x: x * beta_pdf(x), a, b_)
                den, _ = integrate.quad(beta_pdf, a, b_)
                return num / den if den > 1e-15 else (a + b_) / 2

            for _ in range(200):
                boundaries = np.empty(k + 1)
                boundaries[0], boundaries[-1] = -1.0, 1.0
                boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2
                new_c = np.array([cond_mean(boundaries[i], boundaries[i + 1]) for i in range(k)])
                if np.max(np.abs(new_c - centroids)) < 1e-10:
                    break
                centroids = new_c
            centroids = centroids.astype(np.float32)
            boundaries = np.empty(k + 1, dtype=np.float32)
            boundaries[0], boundaries[-1] = -1.0, 1.0
            boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2

            # Generate random unit vectors, quantize, measure MSE
            rng = np.random.default_rng(0)
            vecs = rng.standard_normal((n_samples, dim)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms  # unit sphere

            rotated = vecs @ rotation.T
            indices = np.searchsorted(boundaries[1:-1], rotated).astype(np.uint8)
            y_hat = centroids[indices]
            reconstructed = y_hat @ rotation

            mse = np.mean(np.sum((vecs - reconstructed) ** 2, axis=1))
            ratio_paper = mse / paper_mse[b]
            ratio_lower = mse / lower_bounds[b]
            print(f"{dim:>6} {b:>5} {mse:>10.5f} {paper_mse[b]:>10.3f} {lower_bounds[b]:>10.4f} {ratio_paper:>9.2f}x {ratio_lower:>10.2f}x")

    print()
    print("  Paper Theorem 1: D_mse <= sqrt(3*pi)/2 * 1/4^b  (upper bound)")
    print("  Paper Theorem 3: D_mse >= 1/4^b                 (lower bound, any algorithm)")
    print("  'ours/lower' shows how close we are to the information-theoretic limit.")

    # ================================================================
    # Section 4.4, Figure 5: Recall@1@k on DBpedia
    # ================================================================
    print("\n" + "=" * 80)
    print("Section 4.4, Figure 5: Recall@1@k on DBpedia-OpenAI (d=1536)")
    print("Quantized distance ranking WITHOUT IVF — pure quantizer quality test")
    print("=" * 80)

    base, queries = load_dbpedia(100_000, 1_000)
    print(f"  Base: {base.shape}, Queries: {queries.shape}")

    # Use first 200 queries for speed
    queries = queries[:200]

    print("  Computing ground truth (inner product)...")
    gt = brute_force_knn_ip(base, queries, 100)

    k_values = [1, 2, 4, 8, 16, 32, 64]

    results = {}

    # TurboQuant 4-bit
    print("\n  TurboQuant 4-bit...")
    try:
        t0 = time.time()
        tq4_ranks = tq_quantize_and_rank(base, queries, num_bits=4)
        t_tq4 = time.time() - t0
        results["TQ 4-bit"] = recall_at_1_at_k(tq4_ranks, gt, k_values)
        print(f"    Time: {t_tq4:.1f}s")
    except Exception as e:
        print(f"    Failed: {e}")

    # TurboQuant 2-bit
    print("  TurboQuant 2-bit...")
    try:
        t0 = time.time()
        tq2_ranks = tq_quantize_and_rank(base, queries, num_bits=2)
        t_tq2 = time.time() - t0
        results["TQ 2-bit"] = recall_at_1_at_k(tq2_ranks, gt, k_values)
        print(f"    Time: {t_tq2:.1f}s")
    except Exception as e:
        print(f"    Failed: {e}")

    # PQ (sklearn-based, matches paper setup)
    print("  PQ 4-bit equivalent (M=96, 8-bit codes)...")
    try:
        t0 = time.time()
        pq_ranks = pq_quantize_and_rank(base, queries, num_subvectors=96, num_bits_pq=8)
        t_pq = time.time() - t0
        results["PQ M=96"] = recall_at_1_at_k(pq_ranks, gt, k_values)
        print(f"    Time: {t_pq:.1f}s (includes codebook training)")
    except Exception as e:
        print(f"    Failed: {e}")

    # PQ 2-bit equivalent
    print("  PQ 2-bit equivalent (M=48, 8-bit codes)...")
    try:
        t0 = time.time()
        pq2_ranks = pq_quantize_and_rank(base, queries, num_subvectors=48, num_bits_pq=8)
        t_pq2 = time.time() - t0
        results["PQ M=48"] = recall_at_1_at_k(pq2_ranks, gt, k_values)
        print(f"    Time: {t_pq2:.1f}s")
    except Exception as e:
        print(f"    Failed: {e}")

    # Paper Figure 5 approximate values (read from DBpedia d=1536 plots)
    # These are approximate since we read them from the figure, not exact numbers.
    paper_fig5 = {
        #                k=1    k=2    k=4    k=8   k=16   k=32   k=64
        "TQ 4-bit*":  [0.95,  0.98,  0.99,  1.00,  1.00,  1.00,  1.00],
        "TQ 2-bit*":  [0.84,  0.93,  0.97,  0.99,  1.00,  1.00,  1.00],
        "PQ 4-bit*":  [0.56,  0.74,  0.88,  0.95,  0.98,  1.00,  1.00],
        "PQ 2-bit*":  [0.48,  0.62,  0.74,  0.82,  0.90,  0.95,  0.98],
    }

    # Print results table (matching paper Figure 5 format)
    print(f"\n  Recall@1@k: is the true nearest neighbor in the top-k results?")
    print(f"  (*) = approximate paper values read from Figure 5")
    if results:
        print(f"\n  {'Method':<15}", end="")
        for k in k_values:
            print(f" {'k='+str(k):>7}", end="")
        print()
        print("  " + "-" * (15 + 8 * len(k_values)))

        # Our results
        for method, recalls in results.items():
            print(f"  {method:<15}", end="")
            for k in k_values:
                print(f" {recalls[k]:>7.4f}", end="")
            print()

        # Paper reference values
        print("  " + "-" * (15 + 8 * len(k_values)))
        for method, vals in paper_fig5.items():
            print(f"  {method:<15}", end="")
            for v in vals:
                print(f" {v:>7.2f}", end="")
            print()

    # ================================================================
    # Storage comparison
    # ================================================================
    print("\n" + "=" * 80)
    print("Storage Comparison (d=1536)")
    print("=" * 80)
    dim = 1536
    print(f"\n  {'Method':<25} {'Bytes/vec':>10} {'100K vecs':>10} {'Compression':>12}")
    print("  " + "-" * 60)
    for name, bpv in [
        ("fp32", dim * 4),
        ("fp16", dim * 2),
        ("SQ 8-bit", dim),
        ("TQ 4-bit", dim * 4 // 8 + 4),
        ("TQ 2-bit", dim * 2 // 8 + 4),
        ("TQ 1-bit", dim // 8 + 4),
        ("PQ M=96 8-bit", 96),
        ("PQ M=48 8-bit", 48),
    ]:
        total_mb = bpv * 100_000 / (1024 * 1024)
        comp = f"{dim * 4 / bpv:.1f}x"
        print(f"  {name:<25} {bpv:>10} {total_mb:>8.0f} MB {comp:>12}")

    # ================================================================
    # Quantization time comparison (paper Table 2)
    # ================================================================
    print("\n" + "=" * 80)
    print("Quantization Time (paper Table 2)")
    print("=" * 80)
    print(f"\n  Paper reports training time for 100K vectors at 4-bit quantization:")
    print(f"\n  {'Method':<20} {'d=200':>10} {'d=1536':>10} {'d=3072':>10}")
    print("  " + "-" * 55)
    print(f"  {'PQ':<20} {'37.04s':>10} {'239.75s':>10} {'494.42s':>10}")
    print(f"  {'RabitQ':<20} {'597.25s':>10} {'2267.59s':>10} {'3957.19s':>10}")
    print(f"  {'TurboQuant':<20} {'0.0007s':>10} {'0.0013s':>10} {'0.0021s':>10}")
    print()
    print("  TurboQuant training is 180,000x faster than PQ at d=1536.")
    print("  This is because TQ codebooks are data-oblivious (depend only on d and b).")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
