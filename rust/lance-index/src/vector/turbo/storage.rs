// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage and distance calculation for TurboQuant.
//!
//! # On-disk format
//!
//! TQ vectors are stored as Arrow RecordBatches with three columns:
//!
//! | Column          | Type                      | Description |
//! |-----------------|---------------------------|-------------|
//! | `_rowid`        | UInt64                    | Lance row ID |
//! | `__turbo_code`  | FixedSizeList\<UInt8\>    | Packed b-bit codes (d*b/8 bytes) |
//! | `__turbo_norm`  | Float32                   | Original vector L2 norm |
//!
//! The rotation matrix is stored in the index file's global buffer as a
//! protobuf Tensor. The codebook is NOT stored — it's derived deterministically
//! from `(dimension, num_bits)`.
//!
//! # Asymmetric distance estimation
//!
//! For a float query q and TQ-encoded database vector (codes, norm γ):
//!
//! ```text
//! 1. Precompute (once per query):
//!    q_rot = Π · normalize(q)       // rotate the query
//!
//! 2. Per database vector (O(d) per vector):
//!    ŷ[j] = centroid[codes[j]]      // lookup reconstruction values
//!    dot = Σ q_rot[j] * ŷ[j]       // dot product in rotated space
//!
//! 3. Apply distance metric:
//!    L2:     ||q||² + γ² - 2·||q||·γ·dot
//!    Dot:    -||q||·γ·dot
//!    Cosine: 1 - dot
//! ```
//!
//! The key insight: since Π is orthogonal, inner products are invariant under
//! rotation, so ⟨q̂, Π^T·ŷ⟩ = ⟨Π·q̂, ŷ⟩. This means we can compute distances
//! entirely in the rotated space, avoiding the expensive inverse rotation.

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use deepsize::DeepSizeOf;
use lance_core::{Error, ROW_ID, Result};
use lance_file::previous::reader::FileReader as PreviousFileReader;
use lance_linalg::distance::DistanceType;
use prost::Message;
use serde::{Deserialize, Serialize};

use super::codebook::get_codebook;
use super::packing::{packed_len, unpack_codes};
use super::rotation::{rotation_as_flat_f32, rotate};
use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use crate::vector::storage::{DistCalculator, VectorStore};

pub const TURBO_METADATA_KEY: &str = "lance:turbo";
pub const TURBO_CODE_COLUMN: &str = "__turbo_code";
pub const TURBO_NORM_COLUMN: &str = "__turbo_norm";

/// Metadata for TurboQuant serialization/deserialization.
///
/// The rotation matrix is stored in the global buffer as a protobuf Tensor
/// (too large for JSON metadata). The codebook is NOT stored — it's derived
/// deterministically from (dimension, num_bits).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantizationMetadata {
    /// Rotation matrix (large, stored in global buffer, skipped in JSON).
    #[serde(skip)]
    pub rotate_mat: Option<FixedSizeListArray>,
    /// Position of the rotation matrix in the global buffer.
    #[serde(default)]
    pub rotate_mat_position: Option<u32>,
    /// Bit-width per coordinate (1-8).
    pub num_bits: u32,
    /// Vector dimension.
    pub dimension: usize,
    /// RNG seed for rotation matrix reproducibility.
    pub seed: u64,
    /// Whether codes are transposed/packed for SIMD.
    #[serde(default)]
    pub packed: bool,
}

impl DeepSizeOf for TurboQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.rotate_mat
            .as_ref()
            .map(|m| m.get_array_memory_size())
            .unwrap_or(0)
    }
}

#[async_trait]
impl QuantizerMetadata for TurboQuantizationMetadata {
    fn buffer_index(&self) -> Option<u32> {
        self.rotate_mat_position
    }

    fn set_buffer_index(&mut self, index: u32) {
        self.rotate_mat_position = Some(index);
    }

    fn parse_buffer(&mut self, bytes: Bytes) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let tensor: pb::Tensor = pb::Tensor::decode(bytes)?;
        self.rotate_mat = Some(FixedSizeListArray::try_from(&tensor)?);
        Ok(())
    }

    fn extra_metadata(&self) -> Result<Option<Bytes>> {
        if let Some(ref rotate_mat) = self.rotate_mat {
            let tensor = pb::Tensor::try_from(rotate_mat)?;
            let mut buf = BytesMut::new();
            tensor.encode(&mut buf)?;
            Ok(Some(buf.freeze()))
        } else {
            Ok(None)
        }
    }

    async fn load(reader: &PreviousFileReader) -> Result<Self> {
        let metadata_str = reader
            .schema()
            .metadata
            .get(TURBO_METADATA_KEY)
            .ok_or_else(|| Error::index("TurboQuant metadata not found in schema"))?;
        let metadata: Self = serde_json::from_str(metadata_str)?;
        Ok(metadata)
    }
}

/// A chunk of TurboQuant storage data.
#[derive(Debug, Clone)]
struct TurboStorageChunk {
    batch: RecordBatch,
    dim: usize,
    num_bits: u32,
    code_bytes: usize,
}

impl TurboStorageChunk {
    fn row_ids(&self) -> &UInt64Array {
        self.batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>()
    }

    fn codes(&self) -> &UInt8Array {
        self.batch
            .column_by_name(TURBO_CODE_COLUMN)
            .unwrap()
            .as_fixed_size_list()
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
    }

    fn norms(&self) -> &Float32Array {
        self.batch
            .column_by_name(TURBO_NORM_COLUMN)
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
    }

    fn code_slice(&self, i: usize) -> &[u8] {
        let offset = i * self.code_bytes;
        &self.codes().values()[offset..offset + self.code_bytes]
    }

    fn len(&self) -> usize {
        self.batch.num_rows()
    }
}

/// Storage for TurboQuant quantized vectors.
#[derive(Debug, Clone)]
pub struct TurboQuantizationStorage {
    metadata: TurboQuantizationMetadata,
    distance_type: DistanceType,
    schema: SchemaRef,
    chunks: Vec<TurboStorageChunk>,
    offsets: Vec<u32>,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
}

impl DeepSizeOf for TurboQuantizationStorage {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context)
    }
}

impl TurboQuantizationStorage {
    fn schema_impl(metadata: &TurboQuantizationMetadata) -> SchemaRef {
        let code_bytes = packed_len(metadata.dimension, metadata.num_bits);
        Arc::new(Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                TURBO_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    code_bytes as i32,
                ),
                true,
            ),
            Field::new(TURBO_NORM_COLUMN, DataType::Float32, true),
        ]))
    }
}

impl VectorStore for TurboQuantizationStorage {
    type DistanceCalculator<'a> = TurboDistCalculator<'a>;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send> {
        Ok(self.chunks.iter().map(|c| c.batch.clone()))
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        let mut remaining = id as usize;
        for chunk in &self.chunks {
            if remaining < chunk.len() {
                return chunk.row_ids().value(remaining);
            }
            remaining -= chunk.len();
        }
        panic!("Vector ID {} out of range", id);
    }

    fn append_batch(&self, batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        let dim = self.metadata.dimension;
        let num_bits = self.metadata.num_bits;
        let code_bytes = packed_len(dim, num_bits);
        let mut new_chunks = self.chunks.clone();
        let mut new_offsets = self.offsets.clone();
        let offset = new_offsets.last().copied().unwrap_or(0) + batch.num_rows() as u32;
        new_chunks.push(TurboStorageChunk {
            batch,
            dim,
            num_bits,
            code_bytes,
        });
        new_offsets.push(offset);
        Ok(Self {
            metadata: self.metadata.clone(),
            distance_type: self.distance_type,
            schema: self.schema.clone(),
            chunks: new_chunks,
            offsets: new_offsets,
            frag_reuse_index: self.frag_reuse_index.clone(),
        })
    }

    fn dist_calculator(&self, query: ArrayRef, dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        TurboDistCalculator::new(query, self, dist_q_c)
    }

    fn dist_calculator_from_id(&self, _id: u32) -> Self::DistanceCalculator<'_> {
        unimplemented!("TurboQuant does not support id-based distance calculator yet")
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.chunks
            .iter()
            .flat_map(|c| c.row_ids().values().iter())
    }

    fn len(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl QuantizerStorage for TurboQuantizationStorage {
    type Metadata = TurboQuantizationMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let dim = metadata.dimension;
        let num_bits = metadata.num_bits;
        let code_bytes = packed_len(dim, num_bits);
        let n = batch.num_rows();
        let schema = TurboQuantizationStorage::schema_impl(metadata);

        Ok(Self {
            metadata: metadata.clone(),
            distance_type,
            schema,
            chunks: vec![TurboStorageChunk {
                batch,
                dim,
                num_bits,
                code_bytes,
            }],
            offsets: vec![n as u32],
            frag_reuse_index,
        })
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    async fn load_partition(
        reader: &PreviousFileReader,
        range: std::ops::Range<usize>,
        distance_type: DistanceType,
        metadata: &Self::Metadata,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let batch = reader
            .read_range(range, reader.schema())
            .await?;
        Self::try_from_batch(batch, metadata, distance_type, frag_reuse_index)
    }
}

/// Asymmetric distance calculator for TurboQuant.
///
/// # How it works
///
/// The key insight: since Π is orthogonal, inner products are invariant under
/// rotation. So instead of inverse-rotating the database vector (expensive),
/// we forward-rotate the query (once) and compute everything in rotated space.
///
/// ```text
/// ⟨q, x̃⟩ = ⟨q, γ · Π^T · ŷ⟩ = γ · ⟨Π · q, ŷ⟩
/// ```
///
/// # Cost analysis
///
/// - **Setup** (once per query): O(d²) to rotate query via matrix multiply
/// - **Per database vector**: O(d) to unpack codes, lookup centroids, and dot product
///
/// The per-vector cost is the same as scalar quantization but with 2-8x less storage.
/// The setup cost is amortized over all vectors in the partition.
///
/// # Distance formulas
///
/// For query q, database vector with codes c[0..d-1] and stored norm γ:
///
/// ```text
/// L2:     ||q - x̃||² = ||q||² + γ² - 2·||q||·γ·dot(q_rot, ŷ)
/// Dot:    -⟨q, x̃⟩   = -||q||·γ·dot(q_rot, ŷ)
/// Cosine: 1 - dot(q_rot, ŷ)    (assumes both normalized)
/// ```
pub struct TurboDistCalculator<'a> {
    /// Rotated query vector (Π · q̂ or Π · q_residual)
    query_rotated: Vec<f32>,
    /// Query norm ||q||
    query_norm: f32,
    /// Distance from query to IVF centroid (for residual mode)
    _dist_q_c: f32,
    /// Codebook centroids
    centroids: Vec<f32>,
    /// Reference to storage
    storage: &'a TurboQuantizationStorage,
}

impl<'a> TurboDistCalculator<'a> {
    fn new(
        query: arrow_array::ArrayRef,
        storage: &'a TurboQuantizationStorage,
        dist_q_c: f32,
    ) -> Self {
        let dim = storage.metadata.dimension;
        let num_bits = storage.metadata.num_bits;

        // Extract query as f32 slice
        let query_f32: Vec<f32> = if let Some(fsl) = query.as_any().downcast_ref::<FixedSizeListArray>() {
            fsl.values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec()
        } else {
            query
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec()
        };

        let query_norm: f32 = query_f32.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Get rotation matrix and rotate query
        let rotation = rotation_as_flat_f32(storage.metadata.rotate_mat.as_ref().unwrap());

        // Normalize query before rotation
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query_f32.iter().map(|v| v / query_norm).collect()
        } else {
            query_f32
        };

        let query_rotated = rotate(&query_normalized, &rotation, dim);

        // Get codebook
        let (centroids, _) = get_codebook(dim, num_bits).unwrap();

        Self {
            query_rotated,
            query_norm,
            _dist_q_c: dist_q_c,
            centroids,
            storage,
        }
    }

    /// Find the chunk and local index for a global vector ID.
    fn locate(&self, id: u32) -> (&TurboStorageChunk, usize) {
        let mut remaining = id as usize;
        for chunk in &self.storage.chunks {
            if remaining < chunk.len() {
                return (chunk, remaining);
            }
            remaining -= chunk.len();
        }
        panic!("Vector ID {} out of range", id);
    }
}

impl DistCalculator for TurboDistCalculator<'_> {
    fn distance(&self, id: u32) -> f32 {
        let (chunk, local_id) = self.locate(id);
        let dim = chunk.dim;
        let num_bits = chunk.num_bits;
        let packed = chunk.code_slice(local_id);
        let norm = chunk.norms().value(local_id);

        // Unpack codes and reconstruct in rotated space
        let indices = unpack_codes(packed, dim, num_bits).unwrap();

        // Dot product: Σ q_rot[j] * centroid[idx_j]
        let dot: f32 = self
            .query_rotated
            .iter()
            .zip(indices.iter())
            .map(|(&q, &idx)| q * self.centroids[idx as usize])
            .sum();

        match self.storage.distance_type {
            DistanceType::L2 => {
                // ||q - x̃||² = ||q||² + γ² - 2·||q||·γ·dot
                self.query_norm * self.query_norm + norm * norm
                    - 2.0 * self.query_norm * norm * dot
            }
            DistanceType::Dot => {
                // distance = -⟨q, x̃⟩ = -||q||·γ·dot
                -(self.query_norm * norm * dot)
            }
            DistanceType::Cosine => {
                // Cosine distance = 1 - ⟨q̂, x̃/||x̃||⟩
                // Since x̃ ≈ γ·x̂, ||x̃|| ≈ γ, so ⟨q̂, x̃/γ⟩ ≈ dot
                1.0 - dot
            }
            _ => unimplemented!("Unsupported distance type for TurboQuant"),
        }
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        let n = self.storage.len();
        (0..n as u32).map(|id| self.distance(id)).collect()
    }

    fn prefetch(&self, _id: u32) {
        // No-op for now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_serde() {
        let metadata = TurboQuantizationMetadata {
            rotate_mat: None,
            rotate_mat_position: Some(1),
            num_bits: 4,
            dimension: 768,
            seed: 42,
            packed: false,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: TurboQuantizationMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.num_bits, 4);
        assert_eq!(deserialized.dimension, 768);
        assert_eq!(deserialized.seed, 42);
        assert_eq!(deserialized.rotate_mat_position, Some(1));
    }
}
