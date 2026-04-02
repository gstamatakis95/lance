# IVF_TQ: distributed TurboQuant indexing for Lance

**TurboQuant can be integrated into Lance's generic `IVF<S, Q>` architecture as a new `TurboQuantizer` implementing the `Quantization` trait, requiring zero codebook training, enabling trivially parallel distributed builds at billion scale.** This design document specifies the complete implementation — from Rust trait compliance and on-disk format to Spark-based distributed pipelines and query-path distance estimation. TurboQuant's data-oblivious codebooks (precomputed from the Beta distribution, not learned from data) eliminate the training bottleneck that makes PQ expensive to distribute, while achieving recall superior to PQ, SQ, and RabitQ at equivalent compression ratios. The implementation reuses Lance's existing IVF partitioning, shuffler, and commit infrastructure with minimal changes.

---

## 1. Rust implementation: the `TurboQuantizer` struct

### Quantization trait compliance

Lance's vector index system uses a generic `IVFIndex<S: IvfSubIndex, Q: Quantization>` architecture. The `Quantization` trait (defined in `rust/lance-index/src/vector/quantizer.rs`) requires these methods for any new quantizer:

| Method | Purpose | TurboQuantizer behavior |
|--------|---------|------------------------|
| `quantize_batch(vectors) → RecordBatch` | Encode a batch of float vectors into quantized codes | Apply rotation R, Lloyd-Max scalar quantize each coordinate, pack b-bit codes |
| `compute_distances(query, codes) → Float32Array` | Asymmetric distance from float query to quantized batch | Rotate query once, compute dot/L2 against dequantized coordinates via direct arithmetic |
| `metadata_key() → &str` | Identifier for serialization dispatch | `"turbo"` |
| `column_name() → &str` | Storage column name in auxiliary file | `"__turbo_code"` |
| `use_residual() → bool` | Whether to quantize (vector − centroid) | `true` for L2/Cosine, `false` for Dot |
| Serialize/Deserialize | Write/read metadata to Lance file schema and global buffers | Rotation matrix as protobuf Tensor in global buffer; codebook params + num_bits in JSON |

The existing `Quantizer` enum must gain a new variant:

```rust
pub enum Quantizer {
    Flat,
    Product(ProductQuantizer),
    Scalar(ScalarQuantizer),
    Rabit(RabitQuantizer),
    Turbo(TurboQuantizer),  // NEW
}
```

### Core struct design

```rust
pub struct TurboQuantizer {
    dimension: usize,
    num_bits: u32,                    // 1, 2, 3, 4, or 8
    distance_type: DistanceType,
    rotation_matrix: Float32Array,    // d×d Haar-distributed orthogonal matrix (flattened)
    codebook: Vec<f32>,               // 2^num_bits reconstruction levels for Beta(1/2, (d-1)/2)
    boundaries: Vec<f32>,             // 2^num_bits - 1 decision boundaries
    use_residual: bool,
}

pub struct TurboBuildParams {
    pub num_bits: u32,                // Default: 4
    pub seed: u64,                    // RNG seed for rotation matrix reproducibility
    pub use_structured_rotation: bool,// Use randomized Hadamard instead of dense R
}
```

### Codebook precomputation

The Lloyd-Max codebook for TurboQuant depends **only on dimension d and bit-width b**, not on any data. After rotation by a Haar-distributed orthogonal matrix, each coordinate of a unit-norm vector follows the distribution f(x) = Γ(d/2) / (√π · Γ((d−1)/2)) · (1 − x²)^{(d−3)/2} on [−1, 1], which is a scaled Beta(1/2, (d−1)/2). In high dimensions this converges to N(0, 1/d).

The Lloyd-Max algorithm iteratively solves the continuous 1D k-means problem on this known distribution. **No closed-form solution exists**, but the iteration converges in ~50 steps and needs to run only once per (d, b) pair. For practical embedding dimensions (128, 256, 384, 512, 768, 1024, 1536, 3072) and bit-widths (1, 2, 3, 4, 8), this yields fewer than 40 codebook configurations total.

**Recommended approach**: embed precomputed codebooks as compile-time constants in Rust via a `const` array or `lazy_static!` lookup table keyed by `(dimension, num_bits)`. At b=4 each codebook is just 16 centroids + 15 boundaries = 31 floats (124 bytes). The entire set of all 40 configurations fits in under 5 KB. A `build.rs` script can generate these from a reference Python computation using `scipy.integrate` for the Beta distribution's CDF and Lloyd-Max iteration. For dimensions not in the precomputed set, a runtime fallback computes the codebook in <1 ms.

Representative codebook values (d=768, asymptotically N(0, 1/√768)):

| Bits | Centroids (scaled by √d) | Approx. MSE distortion |
|------|--------------------------|----------------------|
| 1 | ±0.798 | 0.36 |
| 2 | ±0.453, ±1.510 | 0.117 |
| 4 | 16 levels (numerically computed) | 0.009 |

### Rotation matrix generation and storage

The rotation matrix R is generated once via QR decomposition of a d×d matrix with i.i.d. N(0,1) entries. For d=768, R contains **589,824 float32 values (~2.36 MB)**. This is small enough to broadcast to all distributed workers and store in the Lance auxiliary file's global buffer (identical to how RabitQ stores its rotation matrix).

**Structured rotation alternative**: A randomized Hadamard transform (RHT) replaces the dense O(d²) matrix multiply with O(d log d) via the Fast Walsh-Hadamard Transform (FWHT). The RHT applies a random sign-flip diagonal D followed by a Hadamard matrix H: R = HD. This reduces the per-vector transform from **~590K FLOPs to ~7,700 FLOPs** for d=768. The trade-off is that RHT provides approximate (not exact) Haar-distributed rotation, but empirically this has negligible impact on quantization quality. Storage drops from 2.36 MB to just **768 sign bits (96 bytes)** for the diagonal D, since H is implicit.

For the initial implementation, **use dense rotation** for correctness, with a feature flag for structured rotation. The `TurboQuantizer` struct stores either a full matrix or a sign-flip vector depending on the mode:

```rust
pub enum RotationMatrix {
    Dense(Float32Array),         // d×d matrix, ~2.4MB for d=768
    Hadamard { signs: Vec<u8> }, // d sign bits, 96 bytes for d=768
}
```

### Quantization transform (`encode`)

```rust
fn quantize_batch(&self, vectors: &RecordBatch) -> Result<RecordBatch> {
    // For each vector x (or residual x - centroid):
    // 1. Store norm: norm = ||x||₂
    // 2. Normalize: x̂ = x / norm
    // 3. Rotate: y = R · x̂
    // 4. For each coordinate j ∈ [d]:
    //      code[j] = lloyd_max_quantize(y[j], self.boundaries)  // b-bit index
    // 5. Pack codes into ceil(d * b / 8) bytes
    // Output columns: __turbo_code (packed bytes), __turbo_norm (f32)
}
```

**Bit-packing strategy**: Pack b-bit codes sequentially into bytes, little-endian within each byte. For b=4 (the primary target), two codes pack into one byte — **SIMD-friendly** because 4-bit nibble extraction uses simple shift-and-mask. For b=2, four codes per byte. For b=1, eight codes per byte (identical to RabitQ packing). For b=3, codes span byte boundaries; a 3-bit pack/unpack utility handles this with slightly reduced SIMD efficiency.

### Distance estimation

TurboQuant uses **direct arithmetic** rather than lookup tables. The asymmetric distance between a float query q and a quantized database vector x̃ is:

**For inner product**: ⟨q, x̃⟩ = norm_x · ⟨R·q̂, ỹ⟩ where ỹ is the vector of Lloyd-Max reconstruction values for x's codes. The key optimization is to **precompute q_rot = R · q once per query**, then for each database vector, reconstruct ỹ by table lookup (b-bit index → centroid value) and compute the dot product q_rot · ỹ.

**For L2 distance**: ‖q − x‖² = ‖q‖² + norm_x² − 2·norm_x·⟨R·q̂, ỹ⟩. The norms are stored; the inner product is computed as above.

```rust
fn compute_distances(&self, query: &Float32Array, codes: &RecordBatch) -> Result<Float32Array> {
    // 1. Rotate query: q_rot = R · query (once, O(d²) or O(d log d))
    // 2. For each database vector i:
    //    a. Unpack codes[i] → indices[0..d]
    //    b. Reconstruct: for j in 0..d: y_j = codebook[indices[j]]
    //    c. dot = Σ_j q_rot[j] * y_j
    //    d. distance = f(dot, ||query||, norms[i])  // depends on metric
    // 3. Return distances
}
```

**Performance note**: The inner loop (step 2b–c) can be vectorized with SIMD. For 4-bit codes, a **VPSHUFB**-based lookup on x86 AVX2/AVX-512 decodes 32 codes simultaneously and multiplies with the rotated query, achieving throughputs comparable to PQ's lookup-table approach. The reconstruction + dot product costs O(d) per vector, identical to SQ but with **8× better compression** at 4-bit.

---

## 2. On-disk index file format for IVF_TQ

### index.idx — identical to existing IVF indexes

The main index file is unchanged from other IVF variants. It stores the IVF search structure:

**Arrow schema**: `{ __flat_marker: uint64 }` (for FLAT sub-index) or HNSW columns.

**Schema metadata**:
```json
{ "lance:index": "{\"type\": \"IVF_TQ\", \"distance_type\": \"l2\"}" }
{ "lance:ivf": "1" }
{ "lance:flat": "" }
```

**Global buffer [0]**: IVF protobuf message with `centroids_tensor` (num_partitions × dimension float32), `offsets[]`, `lengths[]`, and optional `loss`.

### auxiliary.idx — TurboQuant-specific storage

**Arrow schema**:

| Column | Type | Description |
|--------|------|-------------|
| `_rowid` | `UInt64` | Maps back to original dataset row |
| `__turbo_code` | `FixedSizeList<UInt8>(list_size = ceil(d * b / 8))` | Packed b-bit quantized codes |
| `__turbo_norm` | `Float32` | Original vector L2 norm (needed for distance rescaling) |

For d=768, b=4: `list_size = 768 * 4 / 8 = 384` bytes per vector. For b=2: `list_size = 192`. For b=1: `list_size = 96`.

**Schema metadata**:

| Key | Value |
|-----|-------|
| `"distance_type"` | `"l2"` / `"cosine"` / `"dot"` |
| `"lance:ivf"` | Global buffer index for partition offsets/lengths (no centroids) |
| `"lance:turbo"` | `"{\"version\": 1}"` |
| `"storage_metadata"` | JSON list (see below) |

**`storage_metadata` JSON**:
```json
["{\"rotate_mat_position\": 1, \"num_bits\": 4, \"dimension\": 768, \"use_residual\": true, \"structured_rotation\": false, \"seed\": 42}"]
```

**Global buffers in auxiliary.idx**:
- Buffer [0]: IVF protobuf (offsets + lengths, no centroids)
- Buffer [1]: Rotation matrix as protobuf `Tensor` (shape [d, d], dtype float32) — position referenced by `rotate_mat_position` in metadata

### Comparison with IVF_RQ layout

The IVF_TQ format is structurally nearly identical to IVF_RQ. The differences are:

| Aspect | IVF_RQ | IVF_TQ |
|--------|--------|--------|
| Code column | `_rabit_codes` (d/8 bytes, 1-bit packed) | `__turbo_code` (d·b/8 bytes, b-bit packed) |
| Per-vector scalars | `__add_factors` + `__scale_factors` (2 × f32) | `__turbo_norm` (1 × f32) |
| Metadata key | `"lance:rabit"` | `"lance:turbo"` |
| Codebook storage | None (implicit ±1) | Embedded in binary (compile-time const) |
| Rotation matrix | Global buffer (dense, d×d) | Global buffer (dense d×d or Hadamard signs) |

The reuse of global buffer infrastructure for the rotation matrix, protobuf for IVF metadata, and partition-ordered layout means IVF_TQ can share >90% of the I/O code with IVF_RQ.

---

## 3. Distributed building via IndicesBuilder

### Pipeline mapping

TurboQuant maps cleanly onto Lance's existing 5-step distributed build pipeline. The critical insight is that **TurboQuant eliminates the expensive training step** — the only data-dependent training is IVF k-means, which is already the cheapest step.

```
Step 1: train_ivf()              — UNCHANGED (k-means on sample)
Step 2: train_tq()               — NEW but trivial (generate R, precompute codebook)
Step 3: transform_vectors()      — Reuses existing dispatch with new TQ quantizer
Step 4: shuffle_transformed_vectors() — UNCHANGED (opaque byte arrays)
Step 5: commit_index()           — UNCHANGED (merge shuffled files into final index)
```

### `train_tq()` — the trivial training step

```python
# Python API
tq: TqModel = builder.train_tq(
    num_bits: int = 4,           # Bits per dimension
    seed: int = 42,              # RNG seed for rotation matrix
    structured_rotation: bool = False,  # Use Hadamard instead of dense R
)
```

Internally, this:
1. Generates rotation matrix R via QR decomposition of a `d×d` Gaussian matrix seeded by `seed` (or generates Hadamard sign vector)
2. Looks up the precomputed Lloyd-Max codebook for `(dimension, num_bits)`
3. Returns a `TqModel` wrapping R + codebook + parameters

**No dataset access is needed.** This runs in <1 second for any dimension. Compare with `train_pq()`, which requires sampling and running k-means for each of M subvector spaces (minutes to hours at scale).

### `TqModel` serialization

The `TqModel` must be serializable for distribution to Spark/Ray executors:

```python
class TqModel:
    rotation_matrix: np.ndarray   # shape (d, d) float32, or (d,) int8 for Hadamard signs
    codebook: np.ndarray          # shape (2^b,) float32 — reconstruction levels
    boundaries: np.ndarray        # shape (2^b - 1,) float32 — decision boundaries
    num_bits: int
    dimension: int
    seed: int
    structured_rotation: bool
```

Pickle serialization size: **~2.36 MB** for d=768 dense rotation (dominated by the rotation matrix). With Hadamard: **<1 KB**. Both are small enough for Spark broadcast variables or Ray object store.

### `transform_vectors()` dispatch

The existing `transform_vectors()` method dispatches by quantizer model type. Adding TQ support requires:

```python
# Existing signature — no change needed
builder.transform_vectors(
    ivf: IvfModel,
    quantizer: PqModel | SqModel | TqModel,  # TqModel added to union
    uri: str,
    fragments: List[LanceFragment] = None,
)
```

The Rust backend (`lance_index::vector::ivf::IvfTransformer`) checks the quantizer type and routes to the appropriate encoding path. For TurboQuant, the per-fragment transform:

1. Loads vectors from fragment (Arrow `FixedSizeList<Float32>`)
2. Assigns each vector to nearest IVF partition (unchanged — centroid distance)
3. Computes residual: `r = vector - centroid[partition_id]`
4. Stores norm: `norm = ||r||₂`
5. Normalizes: `r̂ = r / norm`
6. Rotates: `y = R · r̂` (dense matmul or FWHT)
7. Quantizes each coordinate: `code[j] = quantize(y[j], boundaries)`
8. Packs codes into byte array
9. Writes output Lance file: `(partition_id: u32, row_id: u64, __turbo_code: [u8; d*b/8], __turbo_norm: f32)`

**Can this reuse the existing code path?** Partially. Steps 1–2 (partition assignment) are fully reusable. Steps 3–9 require a new `TurboQuantizer::quantize_batch()` implementation, but the outer loop that iterates over fragments, writes intermediate files, and handles I/O is shared. The `IvfTransformer` already dispatches based on `Quantizer` enum variant — adding `Quantizer::Turbo` to the match arm is the only orchestration change.

### Shuffle compatibility

The `shuffle_transformed_vectors()` step sorts intermediate files by `partition_id`. It treats quantized codes as opaque byte arrays — the shuffle has **no knowledge of the quantizer type**. The intermediate file schema includes `partition_id`, `row_id`, and whatever quantized columns the transform produced. Since TurboQuant's output (`__turbo_code` + `__turbo_norm`) is just fixed-size binary + f32, the shuffler handles it identically to PQ codes. No changes needed.

---

## 4. Spark integration at billion scale

### Architecture for 1B × 768-dim fp16 vectors on S3

```
┌─────────────────────── SPARK DRIVER ────────────────────────┐
│                                                               │
│  1. ds = lance.dataset("s3://bucket/embeddings.lance")       │
│  2. builder = IndicesBuilder(ds, "vector")                    │
│  3. ivf = builder.train_ivf(                                  │
│         num_partitions=4096, sample_rate=64, distance_type="l2")│
│  4. tq = builder.train_tq(num_bits=4, seed=42)              │
│  5. Broadcast ivf_bc, tq_bc to executors                      │
│  6. Partition fragments across 64 executors                   │
│                                                               │
│  ┌──────────── 64 SPARK EXECUTORS (mapInArrow) ──────────┐  │
│  │  Load ivf, tq from broadcast                           │  │
│  │  For each assigned fragment:                           │  │
│  │    builder.transform_vectors(ivf, tq, uri, [fragment]) │  │
│  │  Write intermediate Lance files to S3                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  7. Collect transformed file URIs                             │
│  8. Distribute shuffle across executors (optional)            │
│  9. builder.commit_index(shuffled_dir)                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Spark driver code sketch

```python
import lance
from lance.indices import IndicesBuilder
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("IVF_TQ_Build").getOrCreate()
sc = spark.sparkContext

# Step 1-4: Training on driver (< 2 minutes total)
ds = lance.dataset("s3://bucket/embeddings.lance")
builder = IndicesBuilder(ds, "vector")
ivf = builder.train_ivf(num_partitions=4096, sample_rate=64, distance_type="l2")
tq = builder.train_tq(num_bits=4, seed=42)

# Step 5: Broadcast models (~2.4 MB for TQ, ~12 MB for IVF with 4096 partitions)
ivf_bc = sc.broadcast(ivf)
tq_bc = sc.broadcast(tq)

# Step 6: Distribute fragments
fragments = ds.get_fragments()  # ~1000-2000 fragments for 1B vectors
fragment_ids = [f.fragment_id for f in fragments]
output_base = "s3://bucket/tmp/ivf_tq_transform/"

# Step 7: Parallel transform via Spark
fragment_rdd = sc.parallelize(fragment_ids, numSlices=64)

def transform_fragment(frag_id):
    import lance
    from lance.indices import IndicesBuilder
    ds_local = lance.dataset("s3://bucket/embeddings.lance")
    builder_local = IndicesBuilder(ds_local, "vector")
    uri = f"{output_base}frag_{frag_id}"
    fragment = ds_local.get_fragment(frag_id)
    builder_local.transform_vectors(
        ivf_bc.value, tq_bc.value, uri, fragments=[fragment]
    )
    return uri

transformed_uris = fragment_rdd.map(transform_fragment).collect()

# Step 8: Shuffle (can run on driver or distribute)
shuffled_dir = "s3://bucket/tmp/ivf_tq_shuffled/"
builder.shuffle_transformed_vectors(output_base, shuffled_dir)

# Step 9: Commit (single driver operation, ~minutes)
builder.commit_index(shuffled_dir)
```

### Sizing and partitioning for the shuffle phase

At 4-bit with d=768, each quantized vector occupies **388 bytes** (384 code bytes + 4 norm bytes) plus 12 bytes overhead (8 row_id + 4 partition_id) = **400 bytes**. For 1 billion vectors: **~400 GB** of intermediate data.

The shuffle phase must repartition this 400 GB from fragment-order to partition-order. With **4,096 IVF partitions**, each partition averages **~244K vectors (~97 MB)**. The shuffler can process this in streaming fashion, writing one partition at a time. On a single driver node with 64 GB RAM, the shuffler can hold multiple partitions in memory simultaneously.

**For larger scale**, the shuffle can be distributed: split the 4,096 partitions into 64 ranges (64 partitions each), assign each range to an executor, and each executor filters + writes its partition range. This reduces per-executor memory to ~6 GB.

### Rotation matrix broadcast

The dense rotation matrix for d=768 is **2.36 MB** — trivially small for Spark broadcast (limit is typically 8 GB). With Hadamard structured rotation, the broadcast payload drops to 96 bytes for the sign vector. The Hadamard matrix H itself is never stored; it is applied via the FWHT algorithm. **For Spark deployments, Hadamard rotation is strongly recommended** because it eliminates the 2.36 MB broadcast and reduces per-vector compute by ~75×.

### Wall-clock time projections at 1B scale, 64 executors

| Phase | Compute | I/O (S3) | Wall-clock estimate |
|-------|---------|-----------|-------------------|
| train_ivf (driver) | k-means on ~262K sample vectors | Read ~400 MB sample | ~60 seconds |
| train_tq (driver) | QR decomposition of 768×768 matrix | None | <1 second |
| transform_vectors (64 executors) | 15.6M vectors/executor × rotation + quantize | Read ~24 GB/executor, write ~6 GB/executor | **3–5 minutes** (dense R) or **1–2 minutes** (Hadamard) |
| shuffle (driver or distributed) | Sort 400 GB by partition_id | Read + write 400 GB | **8–15 minutes** |
| commit_index (driver) | Build sub-indices, write final files | Write ~400 GB | **5–10 minutes** |
| **Total** | | | **~20–30 minutes** |

For comparison, IVF_PQ at the same scale requires **30–60 minutes for train_pq alone** (k-means on 96 subvector spaces), plus similar transform/shuffle/commit times. TurboQuant's training-free design saves 30–60 minutes on the critical path.

---

## 5. Query path design

### Search pipeline pseudocode

```
SEARCH(query: float[d], k: int, nprobes: int, refine_factor: int):
  
  // Step 1: Partition selection (standard IVF)
  centroid_distances = compute_distances(query, all_centroids)  // O(num_partitions × d)
  top_partitions = argmin_k(centroid_distances, nprobes)
  
  // Step 2: Query preprocessing (once per search)
  IF use_residual:
    FOR each partition p in top_partitions:
      query_residual[p] = query - centroid[p]
      query_norm[p] = ||query_residual[p]||₂
      query_hat[p] = query_residual[p] / query_norm[p]
      query_rot[p] = R · query_hat[p]              // O(d²) or O(d log d)
  ELSE:
    query_rot = R · (query / ||query||)             // Single rotation
  
  // Step 3: Intra-partition scan
  candidates = empty_heap(capacity = k * refine_factor)
  FOR each partition p in top_partitions:
    codes = load_partition_codes(p)                 // __turbo_code column
    norms = load_partition_norms(p)                 // __turbo_norm column
    row_ids = load_partition_rowids(p)              // _rowid column
    
    FOR i in 0..partition_size:
      // Unpack b-bit codes → reconstruction values
      y_hat = dequantize(codes[i], codebook)        // O(d) table lookups
      
      // Asymmetric distance computation
      dot = dot_product(query_rot[p], y_hat)        // O(d) multiply-accumulate
      
      // L2: dist = ||q||² + norm² - 2·||q||·norm·dot
      // IP: dist = -norm * dot
      // Cosine: dist = 1 - dot (if both normalized)
      dist = compute_metric(query_norm[p], norms[i], dot)
      
      candidates.push_if_better(row_ids[i], dist)
  
  // Step 4: Re-ranking with exact vectors (optional)
  IF refine_factor > 1:
    top_candidates = candidates.top(k * refine_factor)
    exact_vectors = read_original_vectors(top_candidates.row_ids)
    reranked = exact_distance(query, exact_vectors)
    RETURN reranked.top(k)
  ELSE:
    RETURN candidates.top(k)
```

### Distance estimation formulas

For a float query q and TurboQuant-encoded database vector with codes c[0..d-1] and stored norm γ:

**L2 distance**:

$$\hat{d}_{L2}(q, x) = \|q_r\|^2 + \gamma^2 - 2\gamma \sum_{j=0}^{d-1} (R \cdot \hat{q}_r)_j \cdot \text{codebook}[c_j]$$

where q_r = q − centroid (residual), q̂_r = q_r / ‖q_r‖, and γ = ‖x_r‖ (stored norm of database residual).

**Inner product**:

$$\widehat{\langle q, x \rangle} = \gamma \sum_{j=0}^{d-1} (R \cdot q)_j \cdot \text{codebook}[c_j]$$

**Cosine similarity**: Normalize both vectors; reduces to inner product on unit vectors.

### QJL residual correction (TurboQuant_prod variant)

The QJL correction adds an unbiased residual estimate to remove the multiplicative bias of MSE-optimal quantizers when estimating inner products. For ANN search, this is **generally unnecessary at ≥3 bits** because the bias is a monotonic scaling factor that preserves ranking order. At 1–2 bits, the bias (2/π ≈ 0.64 at 1-bit) can distort rankings for vectors with similar true distances.

**Recommendation for Lance**: Implement PolarQuant (Stage 1) only for the initial release. The QJL variant requires storing an additional d-bit residual vector and a d×d projection matrix S — doubling storage and compute. It is most valuable for KV cache compression in transformer attention (where unbiased estimates matter) rather than ANN retrieval (where ranking order suffices). A future `IVF_TQ_PROD` variant could add QJL support if demand materializes.

---

## 6. Storage size calculations and performance projections

### Compression ratios at 1B × 768-dim

| Method | Bytes/vector | Total (1B vectors) | Compression vs fp32 | Compression vs fp16 |
|--------|-------------|--------------------|--------------------|---------------------|
| Original fp32 | 3,072 | 3,072 GB | 1× | — |
| Original fp16 | 1,536 | 1,536 GB | 2× | 1× |
| IVF_SQ (8-bit) | 772 | 772 GB | 4× | 2× |
| **IVF_TQ (4-bit)** | **388** | **388 GB** | **8×** | **4×** |
| IVF_PQ (M=48, 8-bit) | 52 | 52 GB | 59× | 30× |
| **IVF_TQ (2-bit)** | **196** | **196 GB** | **16×** | **8×** |
| IVF_RQ (1-bit) | 104 | 104 GB | 30× | 15× |
| **IVF_TQ (1-bit)** | **100** | **100 GB** | **31×** | **15×** |
| IVF_PQ (M=96, 8-bit) | 100 | 100 GB | 31× | 15× |

Note: TQ bytes/vector = ceil(d × b / 8) + 4 (norm). RQ adds 8 bytes (add + scale factors). PQ bytes = M + codebook amortized overhead.

### Recall projections based on TurboQuant paper benchmarks

The paper benchmarks on DBpedia (OpenAI embeddings, d=1536) show TurboQuant consistently outperforming PQ and RabitQ at equivalent bit budgets. Extrapolating to d=768 (typical for modern embedding models):

| Scenario | Expected recall@10 (nprobes=32) | Notes |
|----------|-------------------------------|-------|
| IVF_TQ 4-bit vs IVF_PQ (M=96, 8-bit) | TQ: ~0.95, PQ: ~0.90 | Same 100 bytes/vec; TQ wins by ~5pp |
| IVF_TQ 4-bit vs IVF_SQ 8-bit | TQ: ~0.95, SQ: ~0.97 | SQ uses 2× storage; TQ competitive |
| IVF_TQ 2-bit vs IVF_PQ (M=48, 8-bit) | TQ: ~0.85, PQ: ~0.82 | Similar compression; TQ wins |
| IVF_TQ 1-bit vs IVF_RQ 1-bit | TQ: ~0.70, RQ: ~0.68 | Both 1-bit; TQ slightly better (optimal codebook) |
| IVF_TQ 4-bit vs IVF_TQ 2-bit | 4-bit: ~0.95, 2-bit: ~0.85 | 2× storage → ~10pp recall gain |

These projections assume proper IVF partitioning with nprobes=32 out of 4096 partitions. TurboQuant's advantage over PQ grows with dimension because PQ's subvector independence assumption degrades, while TurboQuant's rotation decorrelates all dimensions optimally.

### Build time comparison

| Operation | TurboQuant | PQ (M=96, 8-bit) | SQ (8-bit) | RabitQ |
|-----------|-----------|-------------------|-----------|--------|
| Training | <1 sec (generate R + lookup codebook) | 30–60 min (96 k-means) | ~10 sec (min/max scan) | ~1 sec (generate R) |
| Per-vector transform (d=768) | ~590K FLOPs (dense R) or ~8K (Hadamard) | ~200K FLOPs | ~1.5K FLOPs | ~590K FLOPs |
| Transform 1B vectors (64 executors) | 3–5 min (dense) / 1–2 min (Hadamard) | ~3 min | ~1 min | ~5 min |
| **End-to-end 1B build** | **~20–30 min** | **~60–90 min** | **~20 min** | **~25 min** |

TurboQuant's end-to-end advantage over PQ is **2–3×** faster, driven entirely by eliminating codebook training. The per-vector transform is actually slower than PQ with dense rotation, but this is parallelizable and amortized across executors.

### Recommended benchmark datasets

- **SIFT1M** (128-dim, 1M vectors): Baseline sanity check; fast iteration
- **GIST1M** (960-dim, 1M vectors): Higher dimensionality stress test
- **GloVe-200** (200-dim, 1.2M vectors): Low-dimensional regime where Beta approximation is less tight
- **Deep1B** (96-dim, 1B vectors): Billion-scale throughput benchmark
- **Synthetic 1B × 768** (768-dim, drawn from unit sphere): Matches production embedding distributions (OpenAI, Cohere, Jina); validates theoretical distortion bounds
- **DBpedia-OpenAI** (1536-dim, 1M vectors): Direct comparison with paper's benchmark

---

## 7. Contribution roadmap

### Phase 1: TurboQuantizer core (Rust) — ~3 weeks, 2 PRs

**PR 1a: Codebook + rotation utilities** (`lance-index` crate)
- `rust/lance-index/src/vector/turbo/codebook.rs` — Lloyd-Max solver for Beta distribution + precomputed codebook table for common (d, b) pairs
- `rust/lance-index/src/vector/turbo/rotation.rs` — Dense rotation via QR decomposition + Hadamard FWHT
- `rust/lance-index/src/vector/turbo/packing.rs` — b-bit pack/unpack with SIMD specializations for b ∈ {1, 2, 4, 8}
- Unit tests: codebook optimality validation (compare MSE against theoretical bound), rotation orthogonality check, pack/unpack round-trip
- **Complexity**: Medium. The Lloyd-Max solver is ~100 lines; FWHT is ~50 lines; bit-packing is ~200 lines with SIMD paths.

**PR 1b: TurboQuantizer struct + Quantization trait impl**
- `rust/lance-index/src/vector/turbo/mod.rs` — `TurboQuantizer` struct, `TurboBuildParams`
- Implement `Quantization` trait: `quantize_batch()`, `compute_distances()`, `metadata_key()`, `column_name()`, `use_residual()`
- Serialization: write rotation matrix as `Tensor` to global buffer, codebook params to `storage_metadata` JSON
- Add `Turbo(TurboQuantizer)` variant to `Quantizer` enum
- Unit tests: encode → decode round-trip MSE matches theoretical bound, distance estimation accuracy vs exact on random vectors
- **Complexity**: Medium-high. The trait impl requires careful alignment with existing patterns from PQ/SQ/RQ implementations.

### Phase 2: IVF_TQ single-node index — ~2 weeks, 1 PR

**PR 2: Wire TurboQuantizer into IVF builder + search**
- Add `IndexType::IvfTq` variant to the index type enum
- Update `IvfIndexBuilder::load_or_build_quantizer()` to handle `TurboBuildParams`
- Update `IvfQuantizationStorage` to read/write TQ auxiliary files
- Update `IVFIndex::search_in_partition()` for TQ distance dispatch
- Integration tests: `create_index(index_type="IVF_TQ")` → search → verify recall on SIFT1M
- **Complexity**: Medium. Most wiring follows established patterns from RabitQ integration.

### Phase 3: Python bindings — ~2 weeks, 1 PR

**PR 3: TqModel + train_tq() + Python API**
- PyO3 wrapper for `TurboQuantizer` → `TqModel` Python class
- `IndicesBuilder.train_tq(num_bits, seed, structured_rotation)` method
- Update `transform_vectors()` to accept `TqModel` (add to type dispatch)
- Pickle support for `TqModel` (serialize rotation matrix + params)
- Python tests: end-to-end `train_ivf → train_tq → create_index → search` workflow
- **Complexity**: Medium. Follows the pattern of existing `PqModel`/`SqModel` bindings.

### Phase 4: IndicesBuilder distributed support — ~1 week, 1 PR

**PR 4: Distributed transform_vectors with TQ**
- Ensure `TqModel` survives serialization across process boundaries (pickle + Ray object store)
- Add `"IVF_TQ"` to the supported types in `IndicesBuilder` validation
- Integration test: multi-process transform → shuffle → commit on a small dataset
- **Complexity**: Low. The distributed pipeline is quantizer-agnostic after Phase 3; this PR is primarily testing and validation.

### Phase 5: lance-ray integration — ~1 week, 1 PR

**PR 5: lance-ray create_index with IVF_TQ**
- Add `"IVF_TQ"` to supported `index_type` literals in `lance_ray.create_index()`
- Pass `num_bits`, `seed`, `structured_rotation` through kwargs to `IndicesBuilder`
- Ray integration test: distributed build on a cluster with ≥4 workers
- **Complexity**: Low. Mostly configuration plumbing + testing.

### Phase 6: Benchmarks and documentation — ~2 weeks, 2 PRs

**PR 6a: Benchmark suite**
- `benchmarks/vector/turbo_benchmark.rs` — Rust micro-benchmarks (encode throughput, distance compute throughput, SIMD vs scalar)
- `python/benchmarks/ivf_tq_recall.py` — Recall@k benchmarks on SIFT1M, GIST1M, DBpedia comparing IVF_TQ vs IVF_PQ vs IVF_SQ vs IVF_RQ
- `python/benchmarks/ivf_tq_build_time.py` — End-to-end build time comparison
- **Complexity**: Medium. Requires dataset downloads and careful measurement methodology.

**PR 6b: Documentation**
- Update `docs/src/indexing.md` with IVF_TQ description, parameter guide, and when-to-use recommendations
- Add IVF_TQ to the LanceDB quantization comparison table
- Jupyter notebook: "Building a billion-vector IVF_TQ index with Spark"
- **Complexity**: Low.

### Total estimated timeline: **10–12 weeks** for a single engineer, with Phases 1–3 on the critical path and Phases 4–6 parallelizable with review cycles.

---

## Conclusion

TurboQuant's data-oblivious design makes it uniquely suited for Lance's distributed index building architecture. Unlike PQ, which requires expensive centralized codebook training before any parallelism begins, TurboQuant's "training" is a sub-second deterministic computation. This eliminates the most significant bottleneck in billion-scale distributed index construction.

Three design decisions deserve emphasis. First, **embed codebooks as compile-time constants** rather than loading from files — the total data is under 5 KB and eliminates a runtime dependency. Second, **implement Hadamard rotation behind a feature flag from day one** — the 75× speedup in the per-vector transform makes it essential for production Spark deployments, even if the initial release uses dense rotation for theoretical purity. Third, **skip QJL residual correction** for the initial release — at ≥3 bits, the MSE-optimal PolarQuant alone provides sufficient ranking accuracy for ANN search, and the QJL variant doubles both storage and computational cost.

The closest existing analog in the codebase is RabitQ (`rust/lance-index/src/vector/bq/`), which shares the random-rotation-then-quantize paradigm. TurboQuant generalizes RabitQ from fixed 1-bit to arbitrary bit-widths with provably optimal codebooks, and the implementation can reuse RabitQ's rotation matrix storage, global buffer infrastructure, and partition I/O patterns almost verbatim. The phased PR plan ensures that each contribution is independently testable and reviewable, with the core quantizer landing first and distributed support layered on top of the existing quantizer-agnostic pipeline.