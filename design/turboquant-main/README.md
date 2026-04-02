# TurboQuant — Reference Implementation & Engineering Insights

> Python implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026) with original engineering findings on KV cache compression for real-world LLMs.

**This is a research companion, not a production tool.** We reproduce the paper's algorithm, integrate it with HuggingFace transformers, and document what we learned — especially what the paper doesn't tell you about making it work on real models.

## Key Findings

### 1. K/V Norm Disparity — The Hidden Obstacle

The paper doesn't discuss this. We found that modern LLMs have dramatically different Key vs Value vector magnitudes:

| Model | K mean norm | V mean norm | Ratio |
|-------|------------|------------|-------|
| GPT-2 (124M) | 11.8 | 2.0 | **6x** |
| Phi-2 (2.8B) | 13.1 | 3.0 | **4x** |
| Qwen2.5-3B | 172.1 | 3.3 | **52x** |
| Qwen2.5-7B | 274.0 | 2.6 | **106x** |
| Qwen2.5-1.5B | 778.6 | 4.3 | **182x** |
| Qwen2.5-0.5B | 259.3 | 0.2 | **1274x** |

Since quantization error scales with norm squared, **K vectors need far more bits than V vectors.** Uniform bit allocation is catastrophically wasteful on models with high K/V ratios.

### 2. MSE > Prod for Attention (Contradicts Paper)

The paper recommends TurboQuantProd for Keys (unbiased inner product) and TurboQuantMSE for Values. Our experiments show **MSE for both is better**:

| GPT-2, b=4 | MSE (both) | Paper (Prod keys) |
|-------------|------------|-------------------|
| PPL change  | +1.1%      | +6.5%             |

**Why:** TurboQuantProd's QJL residual correction adds variance. Softmax attention amplifies variance more than bias. Low variance (MSE) beats unbiasedness (Prod) in practice.

### 3. Outlier-Aware Mixed Precision Closes the Gap

~5-20% of K channels have 10-100x larger RMS than the median (especially Layer 0). Storing these outlier channels at 8-bit while quantizing the rest at 3-bit achieves:

| Method | Avg bits | PPL change (Qwen2.5-1.5B) |
|--------|----------|---------------------------|
| Uniform K=8, V=4 | 6.0 | +0.0% |
| Uniform K=6, V=3 | 4.5 | **+78.1%** |
| **Mixed K=3, V=3** | **3.6** | **+2.1%** |
| Paper target | 3.5 | +0.0% |

Mixed precision at 3.6 bits is within 0.1 bits of the paper's 3.5-bit target.

### 4. K/V Ratio Predicts Optimal Bit Budget

```
K/V ratio < 10x    → 3-bit uniform works      (GPT-2 family)
K/V ratio 10-60x   → 4.5-5 bit asymmetric     (Phi-2, Qwen-3B)
K/V ratio > 100x   → 5.5+ bit or mixed prec.  (Qwen-1.5B, 7B)
K/V ratio > 1000x  → TurboQuant alone insufficient (Qwen-0.5B)
```

### 5. Compressed Storage: 89% Memory Reduction

We implemented actual compressed KV cache storage (not just quantize-dequantize simulation):

| Metric | GPT-2 (41 tokens, 12 layers) |
|--------|------------------------------|
| FP32 KV cache | 2,952 KB |
| Compressed | 327 KB |
| **Reduction** | **89% (9x compression)** |
| PPL impact | None |

## Algorithm Correctness

49 tests pass. MSE distortion matches paper's theoretical bounds:

| Bit-width | Paper theory | Our empirical (d=1536) | Match |
|-----------|-------------|------------------------|-------|
| 1 | 0.360 | 0.363 | 1.01x |
| 2 | 0.117 | 0.117 | 1.00x |
| 3 | 0.030 | 0.035 | 1.15x |
| 4 | 0.009 | 0.009 | 1.05x |

## 8-Model Benchmark

| Model | Params | Best config (<5% PPL) | Avg bits |
|-------|--------|----------------------|----------|
| GPT-2 | 124M | K=3/V=3 | **3.0** |
| GPT-2-Medium | 355M | K=3/V=3 | **3.0** |
| GPT-2-Large | 774M | K=3/V=3 | **3.0** |
| Phi-2 | 2.8B | K=6/V=4 | 5.0 |
| Qwen2.5-3B | 3.1B | K=6/V=3 | 4.5 |
| Qwen2.5-1.5B | 1.5B | K=8/V=4 (or mixed 3/3) | 6.0 (or 3.6) |
| Qwen2.5-7B | 7.6B | K=8/V=3 | 5.5 |

Full results: [AUTO_BENCHMARK_RESULTS.md](AUTO_BENCHMARK_RESULTS.md)

## What This Is NOT

- **Not a production tool** — Pure Python/NumPy, no GPU kernels, too slow for real inference
- **Not the full paper** — Missing PolarQuant, per-channel mixed precision tuning
- **Not a drop-in replacement** — Won't make your local LLM use less VRAM today

## What This IS

- **Correct reference implementation** — Algorithm verified against paper, 49 tests
- **Engineering field notes** — What you actually need to know to make KV cache quantization work
- **Foundation for GPU kernels** — Someone can take these findings and write Triton/CUDA kernels

## Quick Start

```bash
pip install numpy scipy

# Run tests
pytest turboquant/tests/ -v

# Reproduce paper's distortion table
python -m turboquant.benchmarks.distortion

# Run KV cache benchmark on GPT-2
python turboquant/benchmarks/compressed_test.py gpt2
```

## Project Structure

```
turboquant/
├── core.py                  — TurboQuantMSE + TurboQuantProd
├── rotation.py              — Random orthogonal rotation
├── scalar_quantizer.py      — Beta distribution optimal quantizer (Lloyd's algorithm)
├── qjl.py                   — Quantized Johnson-Lindenstrauss (1-bit inner product)
├── mixed_precision.py       — Outlier-aware mixed precision quantizer
├── kv_cache.py              — HuggingFace transformers Cache integration
├── compressed_cache.py      — Actual compressed storage (saves memory)
├── utils.py                 — Normalize, random vectors
├── tests/                   — 49 tests
├── benchmarks/              — Distortion, KV cache, mixed precision benchmarks
├── BENCHMARK_RESULTS.md     — Detailed analysis report
├── AUTO_BENCHMARK_RESULTS.md — 8-model progressive benchmark data
└── TURBOQUANT_STLC_SPECIFICATION.md — STLC spec (used to drive implementation)
```

## Requirements

- Python 3.10+
- NumPy
- SciPy
- pytest (for tests)
- transformers + torch (for KV cache benchmarks)

## Gap to Paper

| Paper claims | Our status | Gap |
|-------------|------------|-----|
| 3.5-bit zero loss | 3.6-bit +2.1% | 0.1 bit, 2.1% quality |
| 8x attention speedup | CPU only (slow) | Need GPU kernel |
| 6x memory reduction | 9x achieved (on CPU) | Need GPU memory management |
| Llama-3.1-8B | Tested up to Qwen2.5-7B | Need Llama access |

## Citation

If you use these findings in your work:

```
@misc{turboquant-ref-impl,
  title={TurboQuant Reference Implementation: Engineering Insights for KV Cache Compression},
  author={Syn-claude and wuko},
  year={2026},
  url={https://github.com/scos-lab/turboquant}
}
```

Paper:
```
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={ICLR},
  year={2026}
}
```

## License

MIT

---

*Built by [scos-lab](https://github.com/scos-lab) — Syn-claude + wuko*
