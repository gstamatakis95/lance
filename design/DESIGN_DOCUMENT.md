# IVF_TQ Design Document: TurboQuant Index for Lance

## Context

We are implementing TurboQuant, a data-oblivious scalar quantizer from the ICLR 2026 paper (Zandieh et al.), as a new vector index type (`IVF_TQ`) in Lance. The algorithm uses random rotation + precomputed Lloyd-Max codebooks on the Beta distribution to achieve near-optimal quantization without any data-dependent training. This makes it uniquely suited for Lance's distributed index building architecture since "training" is a sub-second deterministic computation.

Reference materials:
- `design/2504.19874v1.pdf` — Original paper
- `design/plan.md` — Detailed implementation plan with architecture, formats, and roadmap
- `design/turboquant-main/` — Python reference implementation (core algorithm, tests, benchmarks)

The closest existing analog is RabitQ (`rust/lance-index/src/vector/bq/`), which shares the random-rotation-then-quantize paradigm but is fixed at 1-bit. TurboQuant generalizes to arbitrary bit-widths (1-8) with provably optimal codebooks.

**Important constraint**: Do NOT import from other index modules (bq, sq, pq). Copy needed patterns/code into the turbo module instead, keeping it self-contained.

---

## Algorithm Summary (from Paper)

### TurboQuantMSE (Paper Algorithm 1)

**Goal:** Minimize MSE distortion E[||x - x̃||²] for x ∈ S^{d-1} (unit sphere).

**Global setup (once):**
1. Generate random rotation matrix Π ∈ R^{d×d} via QR decomposition of random Gaussian matrix
2. Construct codebook: find centroids c₁, c₂, ..., c_{2^b} ∈ [-1, 1] that minimize the continuous 1D k-means cost (Eq. 4) under the coordinate distribution induced by rotation

**Quantize(x):**
1. y ← Π · x (rotate to induce Beta-distributed coordinates)
2. idx_j ← arg min_{k∈[2^b]} |y_j - c_k| for every j ∈ [d] (b-bit integer indices)
3. Output: idx

**DeQuantize(idx):**
1. ỹ_j ← c_{idx_j} for every j ∈ [d] (centroid lookup)
2. x̃ ← Π^T · ỹ (inverse rotation)
3. Output: x̃

**Norm extension (paper note after Theorem 1):** For non-unit vectors, store γ = ||x||₂, normalize x̂ = x/γ, apply Algorithm 1 to x̂, rescale reconstruction by γ.

### Coordinate Distribution (Paper Lemma 1)

For x ∈ S^{d-1} (uniform on unit sphere), each coordinate of Π·x follows:

```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}   for x ∈ [-1, 1]
```

This is equivalent to Beta((d-1)/2, (d-1)/2) rescaled from [0,1] to [-1,1]. In high dimensions, this converges to N(0, 1/d).

### Distortion Bounds (Paper Theorem 1)

For any bit-width b ≥ 1 and any x ∈ S^{d-1}:
- D_mse ≤ √(3π)/2 · 1/4^b ≈ 2.7/4^b (within 2.7× of information-theoretic lower bound)
- b=1: D_mse ≈ 0.36 (lower bound: 0.25, ratio: 1.45×)
- b=2: D_mse ≈ 0.117 (lower bound: 0.0625)
- b=3: D_mse ≈ 0.03 (lower bound: 0.0156)
- b=4: D_mse ≈ 0.009 (lower bound: 0.0039)

### TurboQuantProd (Paper Algorithm 2) — DEFERRED

The inner-product-optimal variant applies MSE quantization at (b-1) bits, then QJL (1-bit) on the residual to achieve unbiased inner product estimation. **Deferred for initial release** — for ANN search, MSE-optimal quantization preserves ranking order at ≥3 bits, and QJL doubles storage/compute.

---

## Step 1: Update CLAUDE.md

Add a reference to the `design/` folder in the Architecture section of `CLAUDE.md`.

**File**: `CLAUDE.md` (line 29, after `java/` entry)

```
- `design/` - Design documents and reference implementations for new features
  - `design/turboquant-main/` - Python reference implementation of the TurboQuant algorithm (ICLR 2026)
  - `design/2504.19874v1.pdf` - TurboQuant paper (Zandieh et al.)
  - `design/plan.md` - Implementation plan for the IVF_TQ index type
```

---

## Step 2: Create `rust/lance-index/src/vector/turbo/` module

All code in this module is self-contained — no imports from `bq/`, `sq/`, or `pq/`. Copy patterns where needed.

### 2a: `turbo/mod.rs` — Module root + build params

- Define `TurboBuildParams { num_bits: u32, seed: u64 }` implementing `QuantizerBuildParams` with `sample_size() -> 0`
- Re-export submodules: `codebook`, `rotation`, `packing`, `builder`, `storage`, `transform`

### 2b: `turbo/codebook.rs` — Lloyd-Max codebook precomputation

Faithful port of `design/turboquant-main/scalar_quantizer.py`, verified against paper Algorithm 1, Lemma 1, and Eq. 4.

**Algorithm (Lloyd-Max, solving paper Eq. 4 — continuous 1D k-means on Beta PDF):**
1. Compute alpha = (d-1)/2
2. Initialize centroids via quantile initialization: `quantile_points = linspace(0.5/k, 1 - 0.5/k, k)` where k = 2^b, then map through Beta CDF inverse and rescale from [0,1] to [-1,1]
3. Lloyd iteration (max 200, tolerance 1e-10):
   - Boundaries: midpoints of adjacent centroids; first = -1.0, last = 1.0
   - Centroids: conditional mean E[X | a ≤ X ≤ b] under Beta PDF via numerical integration
   - If denominator CDF < 1e-15, fallback to (a+b)/2
   - Convergence: max |new_centroids - centroids| < 1e-10
4. Output: centroids (f32, len 2^b), boundaries (f32, len 2^b + 1)

**Quantization (equivalent to paper's `arg min |y_j - c_k|`):**
`searchsorted(boundaries[1..-1], value)` → index in [0, k-1]
(Nearest centroid assignment via Voronoi boundaries — boundaries are midpoints of consecutive centroids)

**Dequantization**: `centroids[index]` → f32 reconstruction value

**Paper example centroids (high d, N(0,1/d) approximation):**
- b=1: centroids ≈ ±√(2/π)/√d ≈ ±0.798/√d
- b=2: centroids ≈ ±0.453/√d, ±1.51/√d

**Key numerical constants:**
- Min dimension: d >= 3 (Beta distribution parameter requires this)
- Bit-width: 1-8 (uint8 index dtype, max codebook size 256)
- Lloyd convergence tolerance: 1e-10
- Lloyd max iterations: 200
- Zero-norm threshold: 1e-10
- Conditional mean denominator threshold: 1e-15
- Boundary endpoints: always [-1.0, 1.0]

**Precomputed tables**: Embed compile-time constants for common (d, b) pairs. At b=4 each codebook is 16 centroids + 15 boundaries = 31 floats (124 bytes). Total < 5 KB for all practical configurations.

**Dependencies**: Use `libm` (already in workspace) for gamma/beta functions, or implement numerical integration directly.

### 2c: `turbo/rotation.rs` — Random rotation matrix (self-contained)

Port of `design/turboquant-main/rotation.py`. Copy the QR-based Haar rotation pattern (do NOT import from `bq/`).

**Algorithm (matches paper Section 3.1: "generate Π by applying QR decomposition on a random matrix with i.i.d Normal entries"):**
1. Generate d×d matrix G with i.i.d. N(0,1) entries using seeded RNG
2. QR decomposition: G = Q·R
3. Sign correction: `signs = sign(diag(R))`, replace 0 with 1, then `Q = Q * signs` (column-wise multiply to ensure unique Q with positive diagonal R — Haar-distributed rotation)
4. Store as `FixedSizeListArray` (d rows of d floats)

**Functions:**
- `fn generate_rotation_matrix<T: ArrowFloatType>(dim: usize, seed: u64) -> FixedSizeListArray`
- `fn rotate(vectors: &[f32], rotation: &[f32], dim: usize) -> Vec<f32>` — computes y = Π · x (paper Algorithm 1, line 5)
- `fn inverse_rotate(vectors: &[f32], rotation: &[f32], dim: usize) -> Vec<f32>` — computes x̃ = Π^T · ỹ (paper Algorithm 1, line 10)

**Implementation note:** In row-major storage, `rotate(x)` is implemented as `x @ Π^T` and `inverse_rotate(y)` as `y @ Π`, which are mathematically equivalent to `Π · x` and `Π^T · y` respectively.

**Copy from `bq/builder.rs`**: The `random_orthogonal()` function pattern. Reimplement in turbo module.

### 2d: `turbo/packing.rs` — b-bit pack/unpack

- Pack b-bit codes into byte arrays (little-endian within each byte)
- Specialized paths:
  - b=8: trivial (1 code = 1 byte)
  - b=4: 2 codes per byte (nibble shift-and-mask, SIMD-friendly with VPSHUFB)
  - b=2: 4 codes per byte
  - b=1: 8 codes per byte
  - b=3: codes span byte boundaries (general path, slightly lower SIMD efficiency)

```rust
fn pack_codes(codes: &[u8], num_bits: u32) -> Vec<u8>
fn unpack_codes(packed: &[u8], dim: usize, num_bits: u32) -> Vec<u8>
```

### 2e: `turbo/builder.rs` — `TurboQuantizer` struct + `Quantization` trait impl

**TurboQuantizer fields:** `metadata: TurboQuantizationMetadata`

**Quantization trait impl (implements Paper Algorithm 1 with norm extension):**

Paper Algorithm 1 assumes x ∈ S^{d-1} (unit sphere). For general vectors, we normalize and store the norm separately (paper note after Theorem 1: "compute and store the L2 norms in floating-point precision and rescale the dequantized points using these stored norms").

- `build(data, distance_type, params)` — Generate rotation matrix Π via QR + lookup codebook. Data is ignored (data-oblivious). This is the key advantage: **zero training time** (paper Table 2: 0.0013s for d=1536 vs 239.75s for PQ).
- `quantize(vectors)` — For each vector x:
  1. `γ = ||x||₂` (store norm for rescaling)
  2. `x̂ = x / γ` (normalize to unit sphere; skip if γ < 1e-10)
  3. `y = Π · x̂` (paper Algorithm 1 line 5: rotate to induce Beta-distributed coordinates)
  4. `idx_j = searchsorted(boundaries[1..-1], y_j)` for each j ∈ [d] (paper line 6: nearest centroid via Voronoi boundaries)
  5. `packed = pack_codes(idx, num_bits)` (pack b-bit indices into bytes)
  6. Return packed codes as `FixedSizeList<UInt8>` + norm γ as `Float32`
- Dequantize (for verification/testing):
  1. `ỹ_j = centroids[idx_j]` for each j (paper line 9)
  2. `x̂ = Π^T · ỹ` (paper line 10: inverse rotation)
  3. `x̃ = γ · x̂` (rescale by stored norm)
- `code_dim()` — `ceil(dim * num_bits / 8)`
- `column()` — `"__turbo_code"`
- `metadata_key()` — `"lance:turbo"`
- `quantization_type()` — `QuantizationType::Turbo`
- `use_residual(dt)` — true for L2/Cosine, false for Dot

### 2f: `turbo/storage.rs` — Storage, metadata, distance calculator

**TurboQuantizationMetadata** (serializable, self-contained):
```rust
pub struct TurboQuantizationMetadata {
    #[serde(skip)]
    pub rotate_mat: Option<FixedSizeListArray>,  // d×d rotation matrix (large, stored in global buffer)
    pub rotate_mat_position: Option<u32>,         // Position in global buffer
    pub num_bits: u32,                            // 1-8
    pub dimension: usize,                         // Vector dimension d
    pub seed: u64,                                // RNG seed for reproducibility
    pub packed: bool,                             // Whether codes are bit-packed
}
```
- Codebook is NOT stored in metadata — it's derived from (dimension, num_bits) deterministically
- Rotation matrix stored in global buffer as protobuf Tensor (same pattern as RabitQ but self-contained)

**TurboQuantizationStorage** columns:
- `_rowid` (UInt64) — Maps back to dataset row
- `__turbo_code` (FixedSizeList<UInt8, list_size=ceil(d*b/8)>) — Packed b-bit quantized codes
- `__turbo_norm` (Float32) — Original vector L2 norm

Storage sizes per vector (d=768):
- b=4: 384 code bytes + 4 norm = 388 bytes (8× compression vs fp32)
- b=2: 192 + 4 = 196 bytes (16× compression)
- b=1: 96 + 4 = 100 bytes (31× compression)

**TurboDistCalculator** — Asymmetric distance estimation:

Derived from paper's dequantization (Algorithm 1 lines 9-10): x̃ = γ · Π^T · ỹ

For a float query q and TQ-encoded vector (codes, γ):
1. Precompute (once per query): `q_rot = Π · q̂` where q̂ = q/||q|| (rotate normalized query)
2. Per database vector (codes c[0..d-1], stored norm γ):
   - Reconstruct in rotated space: `ỹ_j = codebook[c_j]` for each j (paper line 9)
   - Dot product in rotated space: `dot = Σ_j q_rot[j] · ỹ_j` (equivalent to ⟨q̂, Π^T·ỹ⟩ by rotation invariance of inner products)
   - **L2**: `||q - x̃||² = ||q||² + γ² - 2·||q||·γ·dot`
   - **Inner product**: `⟨q, x̃⟩ = ||q||·γ·dot` → distance = `-||q||·γ·dot`
   - **Cosine**: `1 - γ·dot/||x̃||` (note: ||x̃|| ≈ γ since ||Π^T·ỹ|| ≈ 1)

Note: When use_residual=true (L2/Cosine), query and database vectors are residuals from IVF centroids. The formulas above apply to the residual vectors.

**Constants:**
```rust
pub const TURBO_METADATA_KEY: &str = "lance:turbo";
pub const TURBO_CODE_COLUMN: &str = "__turbo_code";
pub const TURBO_NORM_COLUMN: &str = "__turbo_norm";
```

### 2g: `turbo/transform.rs` — IVF transform pipeline (self-contained)

Copy the transform pattern from `bq/transform.rs` but do NOT import it:
1. Compute residual: `r = vector - centroid[partition_id]`
2. Store norm: `γ = ||r||₂`
3. Normalize: `r̂ = r / γ`
4. Rotate: `y = Π · r̂`
5. Quantize: `idx_j = searchsorted(boundaries[1..-1], y_j)` for each j
6. Pack codes into bytes
7. Output columns: `__turbo_code`, `__turbo_norm`

---

## Step 3: Wire into existing infrastructure

### 3a: Add `QuantizationType::Turbo` and `Quantizer::Turbo`

**File**: `rust/lance-index/src/vector/quantizer.rs`

- Add `Turbo` to `QuantizationType` enum
- Add `"TQ"` / `"TURBO"` to `FromStr`, `"TQ"` to `Display`
- Add `Turbo(TurboQuantizer)` to `Quantizer` enum
- Add match arms in all `Quantizer` methods: `code_dim()`, `column()`, `metadata_key()`, etc.

### 3b: Add `IndexType::IvfTq`

**File**: `rust/lance-index/src/lib.rs`

- Add `IvfTq = 108` to `IndexType` enum (after `IvfRq = 107`)
- Add `"IVF_TQ"` to `Display`, `TryFrom<i32>`, `TryFrom<&str>` impls
- Add to `is_vector()` match
- Add `IVF_TQ_INDEX_VERSION = 3` constant
- Add to `version()` method

### 3c: Update IVF dispatch

**File**: `rust/lance-index/src/vector/ivf.rs` and related

- Add TurboQuantizer to `new_ivf_transformer_with_quantizer()` dispatch

### 3d: Register module

**File**: `rust/lance-index/src/vector.rs`

- Add `pub mod turbo;`

---

## Step 4: Main `lance` crate integration

**File**: `rust/lance/src/index/vector/builder.rs` (and related)

- Add IVF_TQ to index builder dispatch
- Wire `TurboBuildParams` into `load_or_build_quantizer()`

---

## Step 5: Python bindings

### 5a: `python/src/indices.rs` — PyO3 bindings
- Add `PyTqModel` class
- Add `train_tq_model()` function
- Update `transform_vectors()` to accept TQ model

### 5b: `python/python/lance/indices/builder.py` + `dataset.py`
- Add `TqModel` class with save/load/pickle
- Add `IndicesBuilder.train_tq(num_bits, seed)` method
- Add `"IVF_TQ"` to `create_index()` routing

---

## Step 6: Add README

**File**: `rust/lance-index/src/vector/turbo/README.md`

Contents:
- Overview of TurboQuant algorithm (rotation → quantize → pack)
- Link to paper and reference implementation in `design/`
- Module structure (which file does what)
- Usage example (create IVF_TQ index)
- Key design decisions: data-oblivious codebooks, no QJL initially, dense rotation first
- Numerical details: Beta distribution parameters, Lloyd-Max convergence, bit-packing layout

---

## Step 7: Benchmarks

Four benchmark categories, following existing patterns in `rust/lance-index/benches/` and `python/python/benchmarks/`.

### 7a: Distortion Benchmark (Theorem 1 validation)

Reproduces paper Table 1 / `design/turboquant-main/benchmarks/distortion.py`.

**Rust unit test** in `turbo/codebook.rs` or dedicated `turbo/bench_distortion.rs`:
```
For each (d, b) in {128, 512, 768, 1536} × {1, 2, 3, 4}:
  1. Create TurboQuantizer with seed=42
  2. Generate 5000 random unit vectors (same seed=0 as reference impl)
  3. Quantize → dequantize → compute MSE = mean(||x - x̃||²)
  4. Assert MSE matches paper Theorem 1 bounds:
     - b=1: MSE < 0.50 (paper: ≈0.36)
     - b=2: MSE < 0.18 (paper: ≈0.117)
     - b=3: MSE < 0.05 (paper: ≈0.03)
     - b=4: MSE < 0.015 (paper: ≈0.009)
  5. Print empirical MSE vs paper value and ratio
```

**Expected results** (from reference impl, d=1536):
| b | Paper MSE | Empirical MSE | Ratio |
|---|-----------|---------------|-------|
| 1 | 0.360 | 0.363 | 1.01× |
| 2 | 0.117 | 0.117 | 1.00× |
| 3 | 0.030 | 0.035 | 1.15× |
| 4 | 0.009 | 0.009 | 1.05× |

### 7b: Encode/Decode Throughput Benchmark (Criterion)

**File**: `rust/lance-index/benches/tq.rs` (mirroring `rq.rs` pattern)

```rust
const DIM: usize = 128;  // Also test 768
const TOTAL: usize = 16_000;

fn mock_tq_storage(num_bits: u32) -> TurboQuantizationStorage { ... }

fn construct_dist_table(c: &mut Criterion) {
    for num_bits in [1, 2, 4, 8] {
        // Benchmark: create distance calculator (includes query rotation)
    }
}

fn compute_distances(c: &mut Criterion) {
    for num_bits in [1, 2, 4, 8] {
        // Benchmark: distance_all() and distance(i) for all vectors
    }
}

fn encode_throughput(c: &mut Criterion) {
    for num_bits in [1, 2, 4, 8] {
        // Benchmark: quantize_batch() on TOTAL vectors
    }
}
```

- Measures: distance table construction, batch distance computation, single distance, encoding throughput
- Parametrize over num_bits: {1, 2, 4, 8}
- Uses criterion with 10s measurement time

Reference throughput from Python impl (d=1536, b=3, n=100K): ~50K vec/sec quantize, ~100K vec/sec dequantize. Rust should be 10-50× faster.

**Cargo.toml entry**:
```toml
[[bench]]
name = "tq"
harness = false
```

### 7c: NN Recall Comparison Benchmark

**File**: `python/python/benchmarks/test_tq_recall.py` or extend `test_index.py`

Reproduces paper Section 4.4 / Fig. 5. Compares IVF_TQ against existing quantizers on same dataset.

```python
# Setup: 100K random vectors at d=768, or download DBpedia-OpenAI from HuggingFace
# Ground truth: brute-force exact KNN

@pytest.mark.benchmark
@pytest.mark.parametrize("index_type,params", [
    ("IVF_TQ", {"num_bits": 4}),
    ("IVF_TQ", {"num_bits": 2}),
    ("IVF_PQ", {"num_sub_vectors": 48, "num_bits": 8}),  # ~48 bytes/vec
    ("IVF_SQ", {"num_bits": 8}),                          # ~768 bytes/vec
    ("IVF_RQ", {}),                                        # 1-bit, ~96 bytes/vec
])
def test_recall_comparison(benchmark, index_type, params):
    # Create index, search 1000 queries, measure recall@10
    # Assert TQ recall >= PQ recall at equivalent compression (paper Fig. 5)
```

**Expected results** (from paper Fig. 5, d=1536):
- IVF_TQ 4-bit: recall@1@10 ≈ 0.95
- IVF_TQ 2-bit: recall@1@10 ≈ 0.85
- IVF_PQ (LUT256): recall@1@10 ≈ 0.90 at 4-bit equivalent
- IVF_RQ 1-bit: recall@1@10 ≈ 0.87

### 7d: Build Time Comparison Benchmark

Reproduces paper Table 2. Measures index construction time breakdown.

**File**: `python/python/benchmarks/test_tq_build_time.py`

```python
@pytest.mark.benchmark
@pytest.mark.parametrize("index_type", ["IVF_TQ", "IVF_PQ", "IVF_SQ", "IVF_RQ"])
def test_build_time(benchmark, index_type):
    # 100K vectors, d=768
    # Measure: training time + transform time + total build time
    # TQ training should be <1s (paper Table 2: 0.0013s at d=1536)
    # PQ training should be minutes (paper Table 2: 239.75s at d=1536)
```

**Expected results** (from paper Table 2, 4-bit quantization):
| Method | d=200 | d=1536 | d=3072 |
|--------|-------|--------|--------|
| PQ | 37.04s | 239.75s | 494.42s |
| RabitQ | 597.25s | 2267.59s | 3957.19s |
| **TurboQuant** | **0.0007s** | **0.0013s** | **0.0021s** |

---

## Step 8: Tests

### Unit tests (in turbo module files)

| Test | Location | Validates |
|------|----------|-----------|
| Codebook count | `codebook.rs` | `len(centroids) == 2^b`, `len(boundaries) == 2^b + 1` |
| Codebook symmetry | `codebook.rs` | `centroids[i] + centroids[k-1-i] ≈ 0` for symmetric Beta |
| Boundaries ordered | `codebook.rs` | Strictly increasing, endpoints at -1.0 and 1.0 |
| Centroids in range | `codebook.rs` | All centroids in [-1.0, 1.0] |
| Roundtrip self-quantize | `codebook.rs` | Quantizing centroids returns themselves |
| Distortion vs paper | `codebook.rs` | d=512: b=1 < 0.50, b=2 < 0.18, b=3 < 0.05, b=4 < 0.015 (Theorem 1) |
| Rotation orthogonality | `rotation.rs` | R @ R^T ≈ I (atol 1e-5) |
| Rotation deterministic | `rotation.rs` | Same seed → identical matrix |
| Rotation norm preserve | `rotation.rs` | `\|\|rotate(x)\|\| ≈ \|\|x\|\|` |
| Rotation roundtrip | `rotation.rs` | `inverse_rotate(rotate(x)) ≈ x` |
| Coordinate distribution | `rotation.rs` | After rotation, coordinates follow Beta((d-1)/2, (d-1)/2) (Lemma 1) |
| Pack/unpack roundtrip | `packing.rs` | For b ∈ {1,2,3,4,8}: unpack(pack(codes)) == codes |
| Encode-decode MSE | `builder.rs` | MSE matches paper Theorem 1 distortion bounds |
| Distance accuracy | `storage.rs` | Estimated distance close to exact distance on random vectors |

### Integration tests
- Create IVF_TQ index on random 768-dim vectors → search → recall@10 >= 0.5
- Multi-fragment scenario
- NULL edge cases (null items, all-null, empty collections)

---

## Key Files to Create

| File | Purpose |
|------|---------|
| `rust/lance-index/src/vector/turbo/mod.rs` | Module root, TurboBuildParams |
| `rust/lance-index/src/vector/turbo/codebook.rs` | Lloyd-Max codebook precomputation |
| `rust/lance-index/src/vector/turbo/rotation.rs` | Random rotation matrix (self-contained) |
| `rust/lance-index/src/vector/turbo/packing.rs` | b-bit pack/unpack utilities |
| `rust/lance-index/src/vector/turbo/builder.rs` | TurboQuantizer + Quantization trait impl |
| `rust/lance-index/src/vector/turbo/storage.rs` | Storage, metadata, distance calculator |
| `rust/lance-index/src/vector/turbo/transform.rs` | IVF transform pipeline (self-contained) |
| `rust/lance-index/src/vector/turbo/README.md` | Module documentation |
| `rust/lance-index/benches/tq.rs` | Criterion microbenchmarks (throughput + distance) |
| `python/python/benchmarks/test_tq_recall.py` | NN recall comparison vs PQ/SQ/RQ |
| `python/python/benchmarks/test_tq_build_time.py` | Build time comparison |
| `design/DESIGN_DOCUMENT.md` | This document |

## Key Files to Modify

| File | Change |
|------|--------|
| `CLAUDE.md` | Add `design/` references to Architecture section |
| `rust/lance-index/src/lib.rs` | Add `IvfTq = 108`, `IVF_TQ_INDEX_VERSION = 3` |
| `rust/lance-index/src/vector.rs` | Add `pub mod turbo;` |
| `rust/lance-index/src/vector/quantizer.rs` | Add `Turbo` variant to `QuantizationType` + `Quantizer` enums |
| `rust/lance-index/src/vector/ivf.rs` | Add TQ to transformer dispatch |
| `rust/lance-index/Cargo.toml` | Add `[[bench]] name = "tq"` |
| `rust/lance/src/index/vector/builder.rs` | Add IVF_TQ builder support |
| `python/src/indices.rs` | Add PyTqModel + train_tq_model |
| `python/python/lance/indices/builder.py` | Add TqModel + train_tq method |

## Code to Copy (NOT import) from Other Modules

| What | Source | Destination | Adaptation |
|------|--------|-------------|------------|
| `random_orthogonal()` | `bq/builder.rs` | `turbo/rotation.rs` | Same QR algorithm, new function name |
| Rotation matrix global buffer | `bq/storage.rs` | `turbo/storage.rs` | Same protobuf Tensor pattern |
| IVF transform residual flow | `bq/transform.rs` | `turbo/transform.rs` | Replace 1-bit sign quantization with b-bit Lloyd-Max |
| Storage chunk + DistCalculator | `bq/storage.rs` | `turbo/storage.rs` | Different columns (turbo_code + norm vs rabit_codes + add/scale) |
| Pack sign bits | `bq/builder.rs` | `turbo/packing.rs` | Generalize from 1-bit to arbitrary b-bit |

---

## Verification

1. **Unit tests**: `cargo test -p lance-index turbo`
2. **Codebook correctness**: Compare MSE distortion against paper Theorem 1 bounds
3. **Round-trip**: Quantize → dequantize, verify MSE matches expected distortion per bit-width
4. **Integration**: Create IVF_TQ index on random 768-dim vectors, search, assert recall@10 >= 0.5
5. **Benchmarks**: `cargo bench -p lance-index -- tq` (compare with RQ benchmarks for baseline)
6. **Lint/format**: `cargo clippy --all --tests -- -D warnings && cargo fmt --all`
7. **Python**: `pytest python/python/tests/test_vector_index.py -k turbo`

---

## Paper Reference

Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. arXiv:2504.19874v1.

Key results used in this implementation:
- **Lemma 1**: Coordinate distribution of rotated unit vector → Beta((d-1)/2, (d-1)/2)
- **Algorithm 1**: TurboQuantMSE — rotation + Lloyd-Max scalar quantization
- **Theorem 1**: MSE distortion bound D_mse ≤ √(3π)/2 · 1/4^b
- **Theorem 3**: Information-theoretic lower bound D_mse ≥ 1/4^b
- **Algorithm 2**: TurboQuantProd — MSE + QJL residual (deferred for initial release)
- **Theorem 2**: Inner product distortion bound (deferred)
