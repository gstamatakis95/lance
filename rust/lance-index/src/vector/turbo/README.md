# TurboQuant (`IVF_TQ`) — Data-Oblivious Vector Quantization

Implementation of TurboQuant (Zandieh et al., ICLR 2026) as a Lance vector index type.

## Algorithm

TurboQuant compresses vectors by rotating them with a random orthogonal matrix,
then independently quantizing each coordinate using precomputed optimal codebooks.

```
┌──────────────────────────────────────────────────────────────────┐
│                     TurboQuant Encode Pipeline                   │
│                                                                  │
│  Input x ∈ R^d                                                   │
│     │                                                            │
│     ├─ 1. Store norm: γ = ||x||₂                                 │
│     ├─ 2. Normalize:  x̂ = x / γ           (unit sphere)          │
│     ├─ 3. Rotate:     y = Π · x̂           (Beta-distributed)     │
│     ├─ 4. Quantize:   idx[j] = nearest_centroid(y[j])            │
│     └─ 5. Pack:       codes = pack_b_bit(idx)                    │
│                                                                  │
│  Output: (codes: [u8; d*b/8], norm: f32)                         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    TurboQuant Decode Pipeline                    │
│                                                                  │
│  Input: (codes, norm γ)                                          │
│     │                                                            │
│     ├─ 1. Unpack:     idx = unpack_b_bit(codes)                  │
│     ├─ 2. Lookup:     ŷ[j] = centroid[idx[j]]                    │
│     ├─ 3. Inv rotate: x̂ = Π^T · ŷ                                │
│     └─ 4. Rescale:    x̃ = γ · x̂                                  │
│                                                                  │
│  Output: x̃ ≈ x                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Why it works

After multiplying a unit vector by a random rotation matrix Π, each coordinate
independently follows Beta((d-1)/2, (d-1)/2) on [-1, 1] (paper Lemma 1). Since
we know the exact distribution, we can precompute the optimal scalar quantizer
(Lloyd-Max algorithm) once, offline, for free.

### Key properties

- **Data-oblivious**: Codebooks depend only on (dimension, bit-width), not on data.
  Training takes <1ms. Compare with PQ which needs minutes of k-means.
- **Near-optimal**: Within 2.7x of the information-theoretic lower bound (paper Theorem 1).
- **Parallelizable**: Since training is deterministic, distributed builds at
  billion scale are trivially parallel — no centralized codebook training step.

## Module Structure

```
turbo/
├── mod.rs          TurboBuildParams, module root
├── codebook.rs     Lloyd-Max codebook on Beta distribution
├── rotation.rs     QR-based random orthogonal matrix
├── packing.rs      b-bit pack/unpack (b=1,2,3,4,8)
├── builder.rs      TurboQuantizer + Quantization trait impl
├── storage.rs      On-disk storage, metadata, distance calculator
├── transform.rs    IVF transform pipeline integration
└── README.md       This file
```

### Dependency graph

```
codebook.rs ──────┐
                  ├──→ builder.rs ──→ transform.rs
rotation.rs ──────┤
                  │
packing.rs ───────┘

storage.rs ← uses codebook + rotation + packing for distance calc
```

## Compression Ratios (d=768)

| Bits | Bytes/vector | Compression vs fp32 | Approximate recall@10 |
|------|-------------|---------------------|-----------------------|
| 1    | 100         | 31x                 | ~70%                  |
| 2    | 196         | 16x                 | ~85%                  |
| 4    | 388         | 8x                  | ~95%                  |
| 8    | 772         | 4x                  | ~99%                  |

## Distortion Bounds (paper Theorem 1)

| Bits | MSE distortion | Lower bound | Ratio to optimal |
|------|---------------|-------------|-----------------|
| 1    | 0.36          | 0.25        | 1.45x           |
| 2    | 0.117         | 0.0625      | 1.87x           |
| 3    | 0.03          | 0.0156      | 1.92x           |
| 4    | 0.009         | 0.0039      | 2.31x           |

General bound: D_mse <= sqrt(3*pi)/2 * 1/4^b ≈ 2.7/4^b

## Usage

### Rust

```rust
use lance_index::vector::turbo::{TurboBuildParams, builder::TurboQuantizer};

let params = TurboBuildParams { num_bits: 4, seed: 42 };
let tq = TurboQuantizer::new(768, &params)?;

// Quantize vectors
let codes = tq.quantize(&vectors)?;
```

### Python

```python
import lance
from lance.indices import IndicesBuilder

ds = lance.dataset("my_data.lance")
builder = IndicesBuilder(ds, "vector")

# Train IVF (standard k-means)
ivf = builder.train_ivf(num_partitions=256)

# Train TQ (near-instant, data-oblivious)
tq = builder.train_tq(num_bits=4, seed=42)

# Or use the simple API:
ds.create_index("vector", index_type="IVF_TQ", num_partitions=256, num_bits=4)
```

## Distance Estimation

For a float query q and TQ-encoded vector (codes, norm γ):

1. **Precompute once per query**: rotate the query
   ```
   q_rot = Π · normalize(q)
   ```

2. **Per database vector** (O(d) — same as scalar quantization):
   ```
   ŷ[j] = centroid[codes[j]]           // lookup
   dot = Σ q_rot[j] * ŷ[j]            // dot product
   L2 = ||q||² + γ² - 2·||q||·γ·dot   // distance
   ```

The inner loop can be vectorized with SIMD. For 4-bit codes, VPSHUFB-based
lookup decodes 32 codes simultaneously.

## Design Decisions

1. **Self-contained module**: No imports from `bq/`, `sq/`, or `pq/`.
   Patterns are copied, not shared, to keep the module independent.

2. **Dense rotation first**: QR decomposition produces an exact Haar-distributed
   matrix. Hadamard (FWHT) rotation is a future optimization (75x speedup).

3. **No QJL (TurboQuantProd)**: The inner-product-optimal variant adds a 1-bit
   QJL correction on the residual. For ANN search at >=3 bits, MSE-optimal
   quantization preserves ranking order, so QJL is unnecessary. Deferred.

4. **Codebook caching**: Codebooks are cached in a global `HashMap<(dim, bits)>`
   since they're deterministic. Each codebook is just 31 floats (124 bytes) at b=4.

## Paper Reference

Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025).
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
arXiv:2504.19874v1 [cs.LG]. ICLR 2026.

Key results implemented:
- Algorithm 1 (TurboQuantMSE): rotation + Lloyd-Max scalar quantization
- Theorem 1: MSE distortion bound D_mse <= sqrt(3*pi)/2 * 1/4^b
- Lemma 1: Coordinate distribution Beta((d-1)/2, (d-1)/2) after rotation
