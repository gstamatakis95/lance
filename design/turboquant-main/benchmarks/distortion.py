"""Reproduce TurboQuant paper distortion results (Table 1).

Run: python -m turboquant.benchmarks.distortion
"""

import time

import numpy as np

from turboquant.core import TurboQuantMSE, TurboQuantProd
from turboquant.utils import random_unit_vectors

# Paper theoretical values (Theorem 1 & 2)
PAPER_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
PAPER_IP_VAR = {1: None, 2: 1.57, 3: 0.56, 4: 0.18}  # ×(1/d), b for Prod means total bits


def run_mse_benchmark(dims: list[int], bits: list[int], n_samples: int = 5000):
    print("\n" + "=" * 70)
    print("TurboQuant MSE Distortion Benchmark")
    print(f"Samples per config: {n_samples}")
    print("=" * 70)
    print(f"{'d':>6} {'b':>4} {'MSE_emp':>10} {'MSE_paper':>10} {'ratio':>8} {'time(s)':>8}")
    print("-" * 70)

    for d in dims:
        for b in bits:
            t0 = time.time()
            tq = TurboQuantMSE(d, b, seed=42)
            dist = tq.distortion(n_samples=n_samples, seed=0)
            elapsed = time.time() - t0
            paper_val = PAPER_MSE.get(b, "?")
            if isinstance(paper_val, float):
                ratio = f"{dist / paper_val:.2f}x"
            else:
                ratio = "?"
            print(f"{d:>6} {b:>4} {dist:>10.5f} {str(paper_val):>10} {ratio:>8} {elapsed:>8.2f}")


def run_ip_benchmark(dims: list[int], bits: list[int], n_samples: int = 3000):
    print("\n" + "=" * 70)
    print("TurboQuant Inner Product Distortion Benchmark")
    print(f"Samples per config: {n_samples}")
    print("=" * 70)
    print(f"{'d':>6} {'b':>4} {'bias':>10} {'variance':>10} {'var×d':>10} {'time(s)':>8}")
    print("-" * 70)

    for d in dims:
        for b in bits:
            t0 = time.time()
            tq = TurboQuantProd(d, b, seed=42)
            result = tq.distortion_ip(n_samples=n_samples, seed=0)
            elapsed = time.time() - t0
            var_times_d = result["variance"] * d
            print(
                f"{d:>6} {b:>4} {result['bias']:>10.5f} "
                f"{result['variance']:>10.6f} {var_times_d:>10.4f} {elapsed:>8.2f}"
            )


def run_throughput_benchmark(d: int = 1536, b: int = 3, n: int = 100000):
    print("\n" + "=" * 70)
    print(f"TurboQuant Throughput Benchmark (d={d}, b={b}, n={n:,})")
    print("=" * 70)

    tq = TurboQuantMSE(d, b, seed=42)
    X = random_unit_vectors(n, d, seed=0)

    # Quantize throughput
    t0 = time.time()
    q = tq.quantize(X)
    t_quant = time.time() - t0
    print(f"Quantize:   {n/t_quant:>12,.0f} vec/sec ({t_quant:.3f}s)")

    # Dequantize throughput
    t0 = time.time()
    X_rec = tq.dequantize(q)
    t_dequant = time.time() - t0
    print(f"Dequantize: {n/t_dequant:>12,.0f} vec/sec ({t_dequant:.3f}s)")


if __name__ == "__main__":
    dims = [128, 512, 1536]
    bits = [1, 2, 3, 4]

    run_mse_benchmark(dims, bits)
    run_ip_benchmark(dims, [2, 3, 4])  # b >= 2 for Prod
    run_throughput_benchmark()
