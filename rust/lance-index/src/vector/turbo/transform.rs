// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF transform pipeline for TurboQuant.
//!
//! Integrates TurboQuant into the IVF index building pipeline. This transformer
//! takes residual vectors (vector minus IVF centroid) and produces the quantized
//! representation stored in the index.
//!
//! # Pipeline (within IVF context)
//!
//! The full IVF_TQ transform chain (defined in [`IvfTransformer::with_tq`]):
//!
//! ```text
//! 1. Flatten          → Ensure FixedSizeList format
//! 2. Normalize        → L2 normalize (if Cosine metric)
//! 3. KeepFinite       → Filter NaN/Inf vectors
//! 4. Partition        → Assign vectors to IVF centroids
//! 5. PartitionFilter  → Keep only target partition range
//! 6. Residual         → Subtract centroid: r = x - centroid[part_id]
//! 7. TQTransformer    → Quantize residuals (THIS MODULE)
//! ```
//!
//! The TQTransformer (step 7) takes the residual column and:
//! - Computes norms of residual vectors
//! - Quantizes via TurboQuantizer (normalize → rotate → scalar quantize → pack)
//! - Outputs `__turbo_code` and `__turbo_norm` columns
//! - Drops the original residual vector column

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{Array, Float32Array, RecordBatch};
use arrow_schema::Field;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use tracing::instrument;

use super::builder::TurboQuantizer;
use super::storage::{TURBO_CODE_COLUMN, TURBO_NORM_COLUMN};
use crate::vector::quantizer::Quantization;
use crate::vector::transform::Transformer;

/// Transformer that quantizes residual vectors using TurboQuant.
///
/// Computes norms from the residual vectors before quantization,
/// then produces `__turbo_code` and `__turbo_norm` columns.
pub struct TQTransformer {
    tq: TurboQuantizer,
    vector_column: String,
}

impl TQTransformer {
    pub fn new(tq: TurboQuantizer, vector_column: impl Into<String>) -> Self {
        Self {
            tq,
            vector_column: vector_column.into(),
        }
    }
}

impl Debug for TQTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TQTransformer(vector_column={})", self.vector_column)
    }
}

impl Transformer for TQTransformer {
    #[instrument(name = "TQTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        // If already transformed, skip
        if batch.column_by_name(TURBO_CODE_COLUMN).is_some() {
            return Ok(batch.clone());
        }

        let residual_vectors = batch
            .column_by_name(&self.vector_column)
            .ok_or(Error::index(format!(
                "TQ Transform: column {} not found in batch",
                self.vector_column
            )))?;
        let residual_vectors = residual_vectors
            .as_fixed_size_list_opt()
            .ok_or(Error::index(format!(
                "TQ Transform: column {} is not a fixed size list, got {}",
                self.vector_column,
                residual_vectors.data_type(),
            )))?;

        // Compute norms of residual vectors before quantization
        // (quantize() normalizes internally but we need to store the original norms)
        let values = residual_vectors
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(Error::index(
                "TQ Transform: expected Float32 residual vectors",
            ))?;
        let dim = residual_vectors.value_length() as usize;
        let n = residual_vectors.len();

        let norms: Vec<f32> = (0..n)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                let slice = &values.values()[start..end];
                slice.iter().map(|v| v * v).sum::<f32>().sqrt()
            })
            .collect();
        let norms_array = Float32Array::from(norms);

        // Quantize residual vectors (this normalizes, rotates, and packs internally)
        let tq_codes = self.tq.quantize(residual_vectors)?;

        // Add code and norm columns, drop the residual vector column
        let batch = batch
            .try_with_column(self.tq.field(), tq_codes)
            .map_err(|e| Error::index(e.to_string()))?;
        let batch = batch
            .try_with_column(
                Field::new(TURBO_NORM_COLUMN, arrow_schema::DataType::Float32, true),
                Arc::new(norms_array),
            )
            .map_err(|e| Error::index(e.to_string()))?;

        batch
            .drop_column(&self.vector_column)
            .map_err(|e| Error::index(e.to_string()))
    }
}
