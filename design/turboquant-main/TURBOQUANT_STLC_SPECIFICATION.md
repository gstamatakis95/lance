# TurboQuant Python Implementation — STLC Specification

**Component Name:** TurboQuant — Online Vector Quantization with Near-optimal Distortion Rate
**Version:** 1.0.0
**Specification Type:** STLC (Semantic Tension Language for Code)
**Target Language:** Python 3.10+ (NumPy + optional Triton GPU kernels)
**Last Updated:** 2026-03-27
**Paper:** Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026 (arXiv:2504.19874)
**Reading Notes:** `F:/UoL/books/astren/notes/turboquant-zandieh-2026.md`
**Estimated Total LOC:** ~800 Python (core) + ~400 (tests) + ~200 (benchmarks/demo)
**Total STLC Statements:** ~80

---

## Implementation Guide

**Target File Paths:**
```
turboquant/
├── __init__.py                  (~20 LOC)  — public API exports
├── core.py                      (~250 LOC) — TurboQuantMSE + TurboQuantProd
├── rotation.py                  (~80 LOC)  — random rotation matrix generation
├── scalar_quantizer.py          (~120 LOC) — Beta distribution optimal quantizer
├── qjl.py                       (~80 LOC)  — Quantized Johnson-Lindenstrauss
├── entropy.py                   (~60 LOC)  — optional entropy coding
├── utils.py                     (~40 LOC)  — norm handling, type helpers
├── benchmarks/
│   ├── distortion.py            (~100 LOC) — MSE/inner product distortion measurement
│   └── kv_cache_demo.py         (~100 LOC) — HuggingFace transformers KV cache hook
└── tests/
    ├── test_rotation.py         (~80 LOC)
    ├── test_scalar_quantizer.py (~100 LOC)
    ├── test_core.py             (~150 LOC)
    └── test_qjl.py              (~70 LOC)
```

**Dependencies:**
- `numpy` — 核心计算（rotation, quantization）
- `scipy` — Beta 分布 PDF、最优量化区间求解（scipy.optimize, scipy.special）
- `pytest` — 测试
- `transformers` (optional) — KV cache demo
- `triton` (optional) — GPU kernel，Phase 2

**Implementation Order:**
```
rotation.py → scalar_quantizer.py → qjl.py → core.py → utils.py → entropy.py → tests → benchmarks
```

---

## 1. Overview

### 1.1 Purpose

```stl
[TurboQuant_Impl] → [Purpose] ::mod(
  rule="definitional",
  confidence=1.0,
  description="Implement the TurboQuant algorithm (ICLR 2026) as a standalone Python library. Two quantizers: MSE-optimal (for Value cache) and inner-product-optimal (for Key cache). Data-oblivious, online, near information-theoretic optimality."
)
```

### 1.2 Scope Boundary

```stl
[TurboQuant_Impl] → [Scope_In] ::mod(
  rule="definitional",
  confidence=1.0,
  includes="MSE quantizer, inner product quantizer, QJL, random rotation, optimal scalar quantization, entropy coding, distortion benchmarks, KV cache demo"
)

[TurboQuant_Impl] → [Scope_Out] ::mod(
  rule="definitional",
  confidence=1.0,
  excludes="llama.cpp C integration, Triton GPU kernels (Phase 2), product quantization baseline reimplementation, model fine-tuning"
)
```

### 1.3 Success Criteria

```stl
[TurboQuant_Impl] → [Success_Criteria] ::mod(
  rule="definitional",
  confidence=1.0,
  criterion_1="MSE distortion matches paper Table: b=1→0.36, b=2→0.117, b=3→0.03, b=4→0.009 (±10%)",
  criterion_2="Inner product quantizer is empirically unbiased: |mean bias| < 0.01 over 10k samples",
  criterion_3="Inner product distortion scales as O(1/d) as predicted by Theorem 2",
  criterion_4="Quantization throughput > 100k vectors/sec on CPU for d=1536",
  criterion_5="All tests pass, no external data dependencies"
)
```

---

## 2. Module Specifications

### 2.1 Random Rotation (`rotation.py`)

```stl
[Rotation_Module] → [Purpose] ::mod(
  rule="definitional",
  confidence=1.0,
  description="Generate and apply random orthogonal rotation matrices. The rotation transforms arbitrary unit vectors so that each coordinate follows a known Beta distribution, enabling precomputed optimal quantization."
)

[Rotation_Module] → [Function_generate_rotation] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="generate_rotation(d: int, seed: int | None = None) -> np.ndarray",
  returns="Orthogonal matrix Π ∈ ℝ^(d×d), shape (d, d), dtype float32",
  method="QR decomposition of random Gaussian matrix. Q from qr(randn(d,d)). Ensure det(Q)=+1 (proper rotation).",
  complexity="O(d³) — one-time setup cost",
  note="For d > 4096, use structured rotation (Hadamard + diagonal sign) for O(d log d). Threshold is configurable."
)

[Rotation_Module] → [Function_rotate] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="rotate(x: np.ndarray, rotation: np.ndarray) -> np.ndarray",
  description="y = Π @ x. Supports batched input: x shape (n, d) → y shape (n, d)",
  constraint="x must be on unit sphere (‖x‖₂ = 1). Caller responsible for normalization."
)

[Rotation_Module] → [Function_inverse_rotate] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="inverse_rotate(y: np.ndarray, rotation: np.ndarray) -> np.ndarray",
  description="x̃ = Πᵀ @ y. Πᵀ = Π⁻¹ for orthogonal matrices.",
  note="Π is orthogonal, so inverse = transpose. No matrix inversion needed."
)

[Rotation_Module] → [Structured_Rotation_Optimization] ::mod(
  rule="causal",
  confidence=0.90,
  strength=0.85,
  cause="d > 4096 makes dense rotation O(d²) per vector too expensive",
  effect="Use randomized Hadamard transform: D·H·D' where H is Hadamard, D/D' are random sign diagonals",
  complexity="O(d log d) per vector instead of O(d²)",
  implementation="Phase 1: dense QR only. Add Hadamard path as optimization if benchmarks show bottleneck."
)
```

**Test Requirements:**
```stl
[Test_Rotation] → [Orthogonality] ::mod(
  rule="definitional",
  confidence=1.0,
  test="Π @ Πᵀ ≈ I (Frobenius norm < 1e-6)"
)

[Test_Rotation] → [Norm_Preservation] ::mod(
  rule="definitional",
  confidence=1.0,
  test="‖Π @ x‖₂ ≈ ‖x‖₂ for 1000 random unit vectors (relative error < 1e-6)"
)

[Test_Rotation] → [Coordinate_Distribution] ::mod(
  rule="definitional",
  confidence=0.95,
  test="After rotation, each coordinate of unit vectors follows Beta((d-1)/2, (d-1)/2) rescaled to [-1,1]. Verify with KS test (p > 0.01) for d ∈ {128, 512, 1536}."
)
```

---

### 2.2 Scalar Quantizer (`scalar_quantizer.py`)

```stl
[ScalarQuantizer_Module] → [Purpose] ::mod(
  rule="definitional",
  confidence=1.0,
  description="Compute optimal scalar quantization centroids for the Beta distribution arising from random rotation of unit sphere vectors. This is the continuous 1D k-means problem: given f_X(x) (the Beta PDF), find 2^b centroids minimizing expected squared error."
)

[ScalarQuantizer_Module] → [Beta_PDF] ::mod(
  rule="definitional",
  confidence=1.0,
  formula="f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2), x ∈ [-1, 1]",
  note="For d ≥ 50, well-approximated by 𝒩(0, 1/d). Use exact Beta for correctness."
)

[ScalarQuantizer_Module] → [Function_compute_centroids] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="compute_centroids(d: int, b: int) -> tuple[np.ndarray, np.ndarray]",
  returns="(centroids: array of 2^b values, boundaries: array of 2^b + 1 boundary points)",
  method="Lloyd's algorithm on the continuous Beta distribution. Initialize with uniform quantiles of the Beta CDF. Iterate: (1) update boundaries as midpoints of adjacent centroids, (2) update centroids as conditional means E[X | X ∈ bucket_i]. Converge when centroid shift < 1e-10.",
  complexity="O(2^b · iterations) — precomputed once per (d, b) pair",
  cache="Results are cached per (d, b). Recomputation only on first call."
)

[ScalarQuantizer_Module] → [Function_quantize_scalar] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="quantize_scalar(values: np.ndarray, centroids: np.ndarray, boundaries: np.ndarray) -> np.ndarray",
  returns="Index array (dtype uint8 for b ≤ 8)",
  method="np.searchsorted(boundaries[1:-1], values) — O(n·b) via binary search",
  note="Vectorized over all coordinates of all vectors simultaneously"
)

[ScalarQuantizer_Module] → [Function_dequantize_scalar] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="dequantize_scalar(indices: np.ndarray, centroids: np.ndarray) -> np.ndarray",
  returns="Reconstructed values: centroids[indices]",
  method="Simple lookup table"
)

[ScalarQuantizer_Module] → [Conditional_Mean_Integral] ::mod(
  rule="causal",
  confidence=0.95,
  strength=0.9,
  cause="Lloyd's centroid update requires E[X | a ≤ X ≤ b] under the Beta PDF",
  effect="Use scipy.integrate.quad for numerical integration: ∫_a^b x·f(x)dx / ∫_a^b f(x)dx",
  fallback="For d ≥ 100, Gaussian approximation E[X | a ≤ X ≤ b] under 𝒩(0, 1/d) has closed-form using truncated normal moments"
)
```

**Test Requirements:**
```stl
[Test_ScalarQuantizer] → [Distortion_Matches_Paper] ::mod(
  rule="definitional",
  confidence=1.0,
  test="For d=1536, per-coordinate distortion d·𝒞(f_X, b) matches paper values: b=1→0.36, b=2→0.117, b=3→0.03, b=4→0.009 within ±10%"
)

[Test_ScalarQuantizer] → [Lloyd_Convergence] ::mod(
  rule="definitional",
  confidence=0.98,
  test="Lloyd's algorithm converges within 100 iterations for all b ∈ {1,2,3,4} and d ∈ {128, 512, 1536}"
)

[Test_ScalarQuantizer] → [Centroid_Symmetry] ::mod(
  rule="definitional",
  confidence=1.0,
  test="Centroids are symmetric around 0 (since Beta PDF is symmetric). c_i = -c_{2^b - 1 - i} within 1e-8."
)
```

---

### 2.3 QJL — Quantized Johnson-Lindenstrauss (`qjl.py`)

```stl
[QJL_Module] → [Purpose] ::mod(
  rule="definitional",
  confidence=1.0,
  description="1-bit inner product quantizer using random projection + sign. Provides unbiased inner product estimation with variance O(1/d). Used as the residual corrector in TurboQuant_prod."
)

[QJL_Module] → [Function_generate_projection] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="generate_projection(d: int, seed: int | None = None) -> np.ndarray",
  returns="Random Gaussian matrix S ∈ ℝ^(d×d), i.i.d. 𝒩(0,1) entries",
  note="S is NOT orthogonal — it's a random Gaussian matrix. Different from the rotation Π."
)

[QJL_Module] → [Function_qjl_quantize] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="qjl_quantize(x: np.ndarray, S: np.ndarray) -> np.ndarray",
  returns="sign(S @ x), dtype int8 (+1/-1), shape (d,) or (n, d) for batched",
  description="Random projection followed by sign extraction. Each output bit encodes one random hyperplane comparison."
)

[QJL_Module] → [Function_qjl_dequantize_ip] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="qjl_dequantize_ip(z: np.ndarray, y: np.ndarray, S: np.ndarray, residual_norm: float) -> float",
  returns="Estimated ⟨y, x_residual⟩ = residual_norm · √(π/2) / d · (y @ Sᵀ @ z)",
  description="Reconstruct the inner product contribution from the QJL-quantized residual. NOT a full vector reconstruction — only produces a scalar inner product estimate.",
  math="⟨y, Q_qjl⁻¹(z)⟩ = √(π/2)/d · γ · yᵀ · Sᵀ · z, where γ = ‖residual‖₂"
)

[QJL_Module] → [Unbiasedness_Guarantee] ::mod(
  rule="causal",
  confidence=0.98,
  strength=0.95,
  cause="E[sign(s·x)] = (2/π)·arcsin(x/‖x‖) structure of random hyperplane projections",
  effect="𝔼[⟨y, Q_qjl⁻¹(Q_qjl(x))⟩] = ⟨y, x⟩ exactly",
  source="Lemma 4 in paper, based on Achlioptas (2003) and prior JL literature"
)
```

**Test Requirements:**
```stl
[Test_QJL] → [Unbiasedness] ::mod(
  rule="definitional",
  confidence=1.0,
  test="Over 10,000 random (x, y) pairs on S^(d-1), |mean(estimated_ip - true_ip)| < 0.01 for d ∈ {128, 512, 1536}"
)

[Test_QJL] → [Variance_Bound] ::mod(
  rule="definitional",
  confidence=0.98,
  test="Empirical variance of inner product error ≤ (π/2)/d · ‖y‖² within 20% tolerance"
)
```

---

### 2.4 Core Quantizers (`core.py`)

```stl
[Core_Module] → [Purpose] ::mod(
  rule="definitional",
  confidence=1.0,
  description="Main public API. Two classes: TurboQuantMSE (for Value cache / vector reconstruction) and TurboQuantProd (for Key cache / inner product preservation). Both are stateful — they hold precomputed rotation matrix and quantizer parameters."
)
```

#### 2.4.1 TurboQuantMSE

```stl
[TurboQuantMSE] → [Class_Definition] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="class TurboQuantMSE(d: int, b: int, seed: int | None = None)",
  params="d = vector dimension, b = bit-width per coordinate (1-8), seed = reproducibility",
  state="self.rotation: np.ndarray (d×d), self.centroids: np.ndarray (2^b,), self.boundaries: np.ndarray (2^b+1,)"
)

[TurboQuantMSE] → [Method_quantize] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="quantize(self, x: np.ndarray) -> QuantizedMSE",
  input="x: unit vector(s), shape (d,) or (n, d)",
  output="QuantizedMSE(indices: np.ndarray[uint8], norms: np.ndarray[float32])",
  steps="1. Store norm: γ = ‖x‖₂, normalize x̂ = x/γ. 2. Rotate: y = Π @ x̂. 3. Quantize each coordinate: idx = searchsorted(boundaries, y). 4. Return (idx, γ).",
  note="Non-unit vectors: norm is stored separately and re-applied at dequantize. This extends the algorithm beyond S^(d-1)."
)

[TurboQuantMSE] → [Method_dequantize] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="dequantize(self, q: QuantizedMSE) -> np.ndarray",
  steps="1. Lookup centroids: ỹ = centroids[idx]. 2. Inverse rotate: x̃ = Πᵀ @ ỹ. 3. Rescale: x̃ *= γ.",
  returns="Reconstructed vector(s), same shape as original input"
)

[TurboQuantMSE] → [Method_distortion] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="distortion(self, x: np.ndarray, n_samples: int = 10000) -> float",
  description="Empirical MSE distortion: mean(‖x - dequant(quant(x))‖²₂) over n_samples random unit vectors",
  purpose="Validation against paper's theoretical bounds"
)
```

#### 2.4.2 TurboQuantProd

```stl
[TurboQuantProd] → [Class_Definition] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="class TurboQuantProd(d: int, b: int, seed: int | None = None)",
  params="d = vector dimension, b = total bit-width (allocates b-1 to MSE + 1 to QJL), seed = reproducibility",
  state="self.mse_quantizer: TurboQuantMSE(d, b-1, seed), self.S: np.ndarray (d×d) — QJL projection matrix",
  constraint="b ≥ 2 (need at least 1 bit for MSE + 1 bit for QJL)"
)

[TurboQuantProd] → [Method_quantize] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="quantize(self, x: np.ndarray) -> QuantizedProd",
  output="QuantizedProd(mse_indices: ndarray, qjl_signs: ndarray[int8], residual_norm: float, input_norm: float)",
  steps="1. Store γ = ‖x‖₂, normalize x̂ = x/γ. 2. MSE quantize: q_mse = mse_quantizer.quantize(x̂). 3. Reconstruct: x̃_mse = mse_quantizer.dequantize(q_mse). 4. Residual: r = x̂ - x̃_mse. 5. QJL quantize: signs = sign(S @ r). 6. Store ‖r‖₂. 7. Return all components."
)

[TurboQuantProd] → [Method_dequantize] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="dequantize(self, q: QuantizedProd) -> np.ndarray",
  steps="1. x̃_mse = mse_quantizer.dequantize(q.mse_indices). 2. x̃_qjl = √(π/2)/d · q.residual_norm · Sᵀ @ q.qjl_signs. 3. x̃ = (x̃_mse + x̃_qjl) · q.input_norm.",
  returns="Reconstructed vector (biased for MSE, but UNBIASED for inner products)"
)

[TurboQuantProd] → [Method_inner_product] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="inner_product(self, q: QuantizedProd, y: np.ndarray) -> float",
  description="Compute ⟨y, x⟩ from quantized x without full dequantization. More efficient: avoids reconstructing the full vector.",
  steps="1. ip_mse = ⟨y, dequant_mse(q)⟩. 2. ip_qjl = √(π/2)/d · q.residual_norm · q.input_norm · (y @ Sᵀ @ q.qjl_signs). 3. Return ip_mse + ip_qjl.",
  guarantee="𝔼[return] = ⟨y, x⟩ (unbiased)"
)

[TurboQuantProd] → [Bit_Budget_Allocation] ::mod(
  rule="causal",
  confidence=0.95,
  strength=0.9,
  cause="Total b bits per coordinate must be split between MSE reconstruction and QJL correction",
  effect="b-1 bits → MSE quantizer (reduces residual magnitude), 1 bit → QJL (corrects inner product bias)",
  rationale="QJL is inherently 1-bit (sign function). Giving more bits to MSE reduces residual ‖r‖₂, which reduces QJL variance."
)
```

**Test Requirements:**
```stl
[Test_Core_MSE] → [Distortion_Values] ::mod(
  rule="definitional",
  confidence=1.0,
  test="For d=1536: TurboQuantMSE(d,1).distortion() ≈ 0.36, (d,2) ≈ 0.117, (d,3) ≈ 0.03, (d,4) ≈ 0.009. Tolerance ±15%."
)

[Test_Core_MSE] → [Roundtrip] ::mod(
  rule="definitional",
  confidence=1.0,
  test="quantize → dequantize produces vector of same dimension and approximately same norm"
)

[Test_Core_Prod] → [Unbiasedness] ::mod(
  rule="definitional",
  confidence=1.0,
  test="Over 10,000 random pairs: |mean(estimated_ip - true_ip)| < 0.01 for d=1536, b ∈ {2,3,4}"
)

[Test_Core_Prod] → [Distortion_Scaling] ::mod(
  rule="definitional",
  confidence=0.98,
  test="Inner product distortion scales as O(1/d): doubling d roughly halves distortion"
)

[Test_Core_Prod] → [Bit_Width_Minimum] ::mod(
  rule="definitional",
  confidence=1.0,
  test="TurboQuantProd(d=128, b=1) raises ValueError (need b ≥ 2)"
)
```

---

### 2.5 Data Types (`core.py` — inline)

```stl
[QuantizedMSE] → [Definition] ::mod(
  rule="definitional",
  confidence=1.0,
  type="@dataclass(frozen=True)",
  fields="indices: np.ndarray (uint8), norms: np.ndarray (float32)",
  description="Compressed representation from TurboQuantMSE. indices hold quantizer bin indices per coordinate. norms hold original vector norms for rescaling."
)

[QuantizedProd] → [Definition] ::mod(
  rule="definitional",
  confidence=1.0,
  type="@dataclass(frozen=True)",
  fields="mse_indices: np.ndarray (uint8), qjl_signs: np.ndarray (int8), residual_norm: np.ndarray (float32), input_norm: np.ndarray (float32)",
  description="Compressed representation from TurboQuantProd. Combines MSE quantization indices with QJL sign bits and auxiliary norms."
)

[QuantizedMSE] → [Memory_Layout] ::mod(
  rule="causal",
  confidence=0.95,
  strength=0.9,
  cause="Storage efficiency matters for KV cache compression",
  effect="indices packed as uint8 (supports b ≤ 8). For b ∈ {1,2,4}, bit-packing into bytes is straightforward. For b=3, pack 8 values into 3 bytes (24 bits).",
  phase="Bit-packing is Phase 2 optimization. Phase 1 uses uint8 per coordinate regardless of b."
)
```

---

### 2.6 Entropy Coding (`entropy.py`)

```stl
[Entropy_Module] → [Purpose] ::mod(
  rule="definitional",
  confidence=0.90,
  description="Optional entropy coding exploiting non-uniform centroid assignment probabilities. The Beta distribution assigns different probabilities to each quantizer bin — bins near 0 are more likely than bins near ±1. Entropy coding can save ~5% bits at b=4.",
  priority="Low — implement after core is validated. Nice-to-have, not critical."
)

[Entropy_Module] → [Function_compute_codebook_probs] ::mod(
  rule="definitional",
  confidence=0.95,
  signature="compute_codebook_probs(d: int, b: int, boundaries: np.ndarray) -> np.ndarray",
  returns="Probability of each bin: p_ℓ = ∫_{boundary[ℓ]}^{boundary[ℓ+1]} f_X(x) dx",
  method="scipy.integrate.quad with Beta PDF"
)

[Entropy_Module] → [Function_entropy_bits] ::mod(
  rule="definitional",
  confidence=0.95,
  signature="entropy_bits(probs: np.ndarray) -> float",
  returns="Shannon entropy H = -Σ p_ℓ log₂(p_ℓ)",
  description="Theoretical minimum bits per coordinate with entropy coding. Compare to nominal b bits."
)
```

---

### 2.7 Utilities (`utils.py`)

```stl
[Utils_Module] → [Function_normalize] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]",
  returns="(x_hat, norms) where x_hat = x / ‖x‖₂ and norms = ‖x‖₂",
  description="Separate norm and direction. TurboQuant operates on unit sphere; norms stored separately."
)

[Utils_Module] → [Function_random_unit_vectors] ::mod(
  rule="definitional",
  confidence=1.0,
  signature="random_unit_vectors(n: int, d: int, seed: int | None = None) -> np.ndarray",
  returns="n random unit vectors in ℝᵈ, shape (n, d)",
  method="Sample from 𝒩(0, I_d), then normalize each vector",
  purpose="Test data generation"
)
```

---

## 3. Benchmarks

### 3.1 Distortion Benchmark (`benchmarks/distortion.py`)

```stl
[Benchmark_Distortion] → [Purpose] ::mod(
  rule="definitional",
  confidence=1.0,
  description="Reproduce paper's Table 1 distortion values. Run for d ∈ {128, 512, 1536, 3072}, b ∈ {1,2,3,4}. Report MSE distortion, inner product bias, inner product variance. Compare against theoretical bounds."
)

[Benchmark_Distortion] → [Output_Format] ::mod(
  rule="definitional",
  confidence=1.0,
  format="Markdown table printed to stdout. Columns: d, b, MSE_empirical, MSE_theory, IP_bias, IP_variance_empirical, IP_variance_theory",
  samples="10,000 random unit vectors per (d, b) configuration"
)
```

### 3.2 KV Cache Demo (`benchmarks/kv_cache_demo.py`)

```stl
[Benchmark_KVCache] → [Purpose] ::mod(
  rule="definitional",
  confidence=0.90,
  description="Demonstrate TurboQuant compressing a real model's KV cache. Hook into HuggingFace transformers model, intercept KV cache tensors, quantize Keys with TurboQuantProd and Values with TurboQuantMSE, measure perplexity change.",
  dependency="transformers, torch — optional, not required for core library"
)

[Benchmark_KVCache] → [Target_Model] ::mod(
  rule="definitional",
  confidence=0.85,
  model="Any HuggingFace causal LM that fits in available GPU/CPU memory. Suggest: GPT-2 (small, easy to test) or Llama-3.2-1B (realistic).",
  metric="Perplexity on WikiText-2 or similar standard benchmark"
)
```

---

## 4. Design Decisions

```stl
[Design] → [NumPy_First] ::mod(
  rule="causal",
  confidence=0.95,
  strength=0.9,
  cause="Correctness and readability over raw speed in Phase 1",
  effect="Core implementation in pure NumPy. No Triton/CUDA in Phase 1.",
  rationale="Verify algorithm correctness against paper results first. Optimize later."
)

[Design] → [Stateful_Quantizers] ::mod(
  rule="causal",
  confidence=0.95,
  strength=0.9,
  cause="Rotation matrix and centroids are expensive to compute but reused for every quantize() call",
  effect="TurboQuantMSE and TurboQuantProd are stateful classes, not pure functions. Setup cost is paid once in __init__.",
  tradeoff="Slightly less functional purity, but matches real usage pattern (create quantizer once, quantize millions of vectors)"
)

[Design] → [Non_Unit_Vector_Support] ::mod(
  rule="causal",
  confidence=0.95,
  strength=0.85,
  cause="Real KV cache vectors are not on the unit sphere",
  effect="Quantizers store norms separately and operate on normalized vectors internally. This is a simple extension: factor out ‖x‖₂, quantize x/‖x‖₂, store ‖x‖₂ as float32.",
  cost="4 extra bytes per vector (float32 norm)"
)

[Design] → [No_Bit_Packing_Phase1] ::mod(
  rule="causal",
  confidence=0.90,
  strength=0.85,
  cause="Bit-packing for b=3 (non-power-of-2) is fiddly and obscures algorithm logic",
  effect="Phase 1 stores each coordinate as uint8 regardless of b. Actual compression ratio is worse than theoretical. Phase 2 adds proper bit-packing.",
  note="This means Phase 1 memory savings are limited to demonstrating the algorithm, not achieving production compression ratios."
)
```

---

## 5. Implementation Order & Dependency Graph

```stl
[Step_1] → [rotation.py] ::mod(
  rule="definitional",
  confidence=1.0,
  description="Random rotation generation and application. No dependencies.",
  est_loc=80,
  test="test_rotation.py"
)

[Step_2] → [scalar_quantizer.py] ::mod(
  rule="definitional",
  confidence=1.0,
  description="Beta PDF + Lloyd's algorithm + quantize/dequantize. Depends on scipy.",
  est_loc=120,
  test="test_scalar_quantizer.py"
)

[Step_3] → [qjl.py] ::mod(
  rule="definitional",
  confidence=1.0,
  description="QJL random projection + sign quantization + inner product estimation. No internal dependencies.",
  est_loc=80,
  test="test_qjl.py"
)

[Step_4] → [utils.py] ::mod(
  rule="definitional",
  confidence=1.0,
  description="normalize(), random_unit_vectors(). No dependencies.",
  est_loc=40
)

[Step_5] → [core.py] ::mod(
  rule="definitional",
  confidence=1.0,
  description="TurboQuantMSE + TurboQuantProd + data types. Depends on Steps 1-4.",
  est_loc=250,
  test="test_core.py"
)

[Step_6] → [entropy.py] ::mod(
  rule="definitional",
  confidence=0.90,
  description="Optional entropy coding. Depends on scalar_quantizer.py.",
  est_loc=60,
  priority="low"
)

[Step_7] → [benchmarks/distortion.py] ::mod(
  rule="definitional",
  confidence=1.0,
  description="Reproduce paper results. Depends on core.py.",
  est_loc=100
)

[Step_8] → [benchmarks/kv_cache_demo.py] ::mod(
  rule="definitional",
  confidence=0.85,
  description="KV cache compression demo. Depends on core.py + transformers.",
  est_loc=100,
  priority="medium"
)
```

```
rotation.py ──────┐
                   ├──→ core.py ──→ benchmarks/distortion.py
scalar_quantizer.py┤                      │
                   │               benchmarks/kv_cache_demo.py
qjl.py ───────────┤
                   │
utils.py ─────────┘

entropy.py ← scalar_quantizer.py (optional)
```

---

## 6. Future Phases (Out of Scope)

```stl
[Phase_2] → [Triton_GPU_Kernels] ::mod(
  rule="definitional",
  confidence=0.80,
  description="Rewrite quantize/dequantize as Triton kernels for GPU inference. Target: integrate with vLLM or SGLang.",
  trigger="Phase 1 correctness validated + demand exists"
)

[Phase_3] → [llama_cpp_Integration] ::mod(
  rule="definitional",
  confidence=0.70,
  description="C/C++ implementation for llama.cpp integration. Would enable TurboQuant in Ollama/koboldcpp ecosystem.",
  trigger="Phase 2 GPU performance validated + community interest"
)

[Phase_4] → [scos_lab_Release] ::mod(
  rule="definitional",
  confidence=0.75,
  description="Open source release under scos-lab GitHub. PyPI package. Paper citation. Benchmarks against KIVI/PolarQuant.",
  trigger="Phase 1 complete + benchmarks reproduced"
)
```

---

*Specification complete: 2026-03-27 | Syn-claude*
*Paper: Zandieh et al., TurboQuant, ICLR 2026 (arXiv:2504.19874)*
