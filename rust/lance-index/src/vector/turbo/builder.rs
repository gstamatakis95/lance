// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! TurboQuantizer: implements the [`Quantization`] trait for TurboQuant.
//!
//! Implements Paper Algorithm 1 (TurboQuantMSE) with norm extension for
//! non-unit vectors. The quantizer is data-oblivious: [`build()`](TurboQuantizer::build)
//! ignores the input data and generates the rotation matrix + codebook
//! deterministically from `(dimension, num_bits, seed)`.
//!
//! # Quantization pipeline (per vector)
//!
//! ```text
//! 1. γ = ||x||₂                           Store norm
//! 2. x̂ = x / γ                            Normalize to unit sphere
//! 3. y = Π · x̂                            Rotate (induces Beta coords)
//! 4. idx[j] = searchsorted(boundaries, y[j])  Scalar quantize each coord
//! 5. packed = pack_b_bit(idx)              Pack into bytes
//! ```
//!
//! # Dequantization pipeline (per vector)
//!
//! ```text
//! 1. idx = unpack_b_bit(packed)            Unpack codes
//! 2. ŷ[j] = centroid[idx[j]]              Centroid lookup
//! 3. x̂ = Π^T · ŷ                          Inverse rotation
//! 4. x̃ = γ · x̂                            Rescale by norm
//! ```
//!
//! # Data-oblivious advantage
//!
//! Unlike PQ which requires expensive k-means training on sampled data,
//! TurboQuantizer's `build()` method ignores the input entirely. The codebook
//! is a deterministic function of `(dim, num_bits)`, and the rotation matrix
//! is a deterministic function of `(dim, seed)`. This means:
//!
//! - Training time: <1ms (vs minutes/hours for PQ at large scale)
//! - No data sampling needed (`sample_size() = 0`)
//! - Trivially parallelizable for distributed index builds

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::Float32Type;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, UInt8Array};
use arrow_schema::{DataType, Field};
use deepsize::DeepSizeOf;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};

use super::codebook::{get_codebook, quantize_scalar};
use super::packing::{pack_codes, packed_len};
use super::rotation::{generate_rotation_matrix, rotate, rotation_as_flat_f32};
use super::TurboBuildParams;
use crate::vector::quantizer::{Quantization, QuantizationMetadata, Quantizer};
use crate::vector::turbo::storage::{
    TurboQuantizationMetadata, TurboQuantizationStorage, TURBO_CODE_COLUMN, TURBO_METADATA_KEY,
    TURBO_NORM_COLUMN,
};

/// TurboQuantizer: data-oblivious vector quantizer (paper Algorithm 1).
///
/// The quantizer state consists of:
/// - **Rotation matrix Π** (d×d): Random orthogonal matrix generated via
///   QR decomposition of a seeded Gaussian matrix. Stored in the index file's
///   global buffer as a protobuf Tensor.
/// - **Codebook**: Lloyd-Max optimal centroids for Beta((d-1)/2, (d-1)/2).
///   NOT stored — derived deterministically from `(dimension, num_bits)`.
///
/// # Data-oblivious property
///
/// Unlike PQ which requires k-means training on sampled data, TurboQuantizer
/// ignores the input data entirely. The [`build()`](Self::build) method only
/// reads the dimension from the input array. This means:
///
/// - **Training**: <1ms (vs minutes/hours for PQ)
/// - **Sample size**: 0 (no data needed)
/// - **Deterministic**: Same `(dim, num_bits, seed)` always produces the same quantizer
/// - **Parallelizable**: Workers can independently create identical quantizers
///
/// # Storage per vector
///
/// | bits | d=128 | d=768 | d=1536 |
/// |------|-------|-------|--------|
/// | 1    | 20 B  | 100 B | 196 B  |
/// | 2    | 36 B  | 196 B | 388 B  |
/// | 4    | 68 B  | 388 B | 772 B  |
/// | 8    | 132 B | 772 B | 1540 B |
///
/// Formula: ceil(d × b / 8) + 4 bytes (norm)
#[derive(Debug, Clone, DeepSizeOf)]
pub struct TurboQuantizer {
    pub(crate) metadata: TurboQuantizationMetadata,
}

impl TurboQuantizer {
    /// Create a new TurboQuantizer for the given dimension and parameters.
    ///
    /// This is near-instant (<1ms) since it only:
    /// 1. Generates a d×d rotation matrix via QR decomposition of a seeded Gaussian matrix
    /// 2. Looks up (or computes) the Lloyd-Max codebook for `(dim, num_bits)`
    ///
    /// # Errors
    ///
    /// Returns an error if `dim < 3` or `num_bits` is not in `1..=8`.
    pub fn new(dim: usize, params: &TurboBuildParams) -> Result<Self> {
        if dim < 3 {
            return Err(Error::invalid_input(format!(
                "TurboQuant requires dimension >= 3, got {}",
                dim
            )));
        }
        if params.num_bits == 0 || params.num_bits > 8 {
            return Err(Error::invalid_input(format!(
                "TurboQuant num_bits must be 1-8, got {}",
                params.num_bits
            )));
        }

        let rotate_mat = generate_rotation_matrix::<Float32Type>(dim, params.seed);

        let metadata = TurboQuantizationMetadata {
            rotate_mat: Some(rotate_mat),
            rotate_mat_position: None,
            num_bits: params.num_bits,
            dimension: dim,
            seed: params.seed,
            packed: false,
        };

        Ok(Self { metadata })
    }

    /// Get the rotation matrix as a FixedSizeListArray reference.
    pub fn rotation_matrix(&self) -> Option<&FixedSizeListArray> {
        self.metadata.rotate_mat.as_ref()
    }

    /// Get the bit-width per coordinate.
    pub fn num_bits(&self) -> u32 {
        self.metadata.num_bits
    }

    /// Get the RNG seed.
    pub fn seed(&self) -> u64 {
        self.metadata.seed
    }

    /// Get the rotation matrix as a flat f32 slice.
    fn rotation_flat(&self) -> Vec<f32> {
        rotation_as_flat_f32(self.metadata.rotate_mat.as_ref().unwrap())
    }

    /// Get the codebook for the current configuration.
    fn codebook(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        get_codebook(self.metadata.dimension, self.metadata.num_bits)
    }

    /// Quantize float32 vectors using paper Algorithm 1 (TurboQuantMSE).
    ///
    /// # Algorithm (per vector x)
    ///
    /// 1. **Store norm**: `γ = ||x||₂`
    /// 2. **Normalize**: `x̂ = x / γ` (map to unit sphere S^{d-1})
    /// 3. **Rotate**: `y = Π · x̂` (induces Beta-distributed coordinates, Lemma 1)
    /// 4. **Scalar quantize**: `idx[j] = searchsorted(boundaries, y[j])` for each j
    /// 5. **Pack**: Bit-pack the b-bit indices into bytes
    ///
    /// # Returns
    ///
    /// `FixedSizeList<UInt8>` of packed codes. Each row has `ceil(d × b / 8)` bytes.
    /// The norms are computed and stored separately by the IVF transform pipeline.
    ///
    /// # Performance
    ///
    /// The dominant cost is the matrix multiply in step 3: O(d²) per vector with
    /// dense rotation. This is the same cost as RabitQ. With Hadamard rotation
    /// (future optimization), this drops to O(d log d).
    fn quantize_f32(&self, vectors: &FixedSizeListArray) -> Result<ArrayRef> {
        let dim = self.metadata.dimension;
        let num_bits = self.metadata.num_bits;
        let n = vectors.len();
        let code_bytes = packed_len(dim, num_bits);

        let rotation = self.rotation_flat();
        let (_centroids, boundaries) = self.codebook()?;

        let values = vectors
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| Error::invalid_input("Expected Float32 vectors"))?;
        let values_slice = values.values();

        let mut all_codes = Vec::with_capacity(n * code_bytes);

        for i in 0..n {
            let x = &values_slice[i * dim..(i + 1) * dim];

            // Step 1: compute norm
            let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

            // Step 2: normalize to unit sphere
            let x_hat: Vec<f32> = if norm < 1e-10 {
                x.to_vec()
            } else {
                x.iter().map(|v| v / norm).collect()
            };

            // Step 3: rotate (y = Π · x̂)
            let y = rotate(&x_hat, &rotation, dim);

            // Step 4: quantize each coordinate
            let indices: Vec<u8> = y
                .iter()
                .map(|&val| quantize_scalar(val, &boundaries))
                .collect();

            // Step 5: pack codes
            let packed = pack_codes(&indices, num_bits)?;
            all_codes.extend_from_slice(&packed);
        }

        let codes_array = UInt8Array::from(all_codes);
        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            codes_array,
            code_bytes as i32,
        )?))
    }
}

impl Quantization for TurboQuantizer {
    type BuildParams = TurboBuildParams;
    type Metadata = TurboQuantizationMetadata;
    type Storage = TurboQuantizationStorage;

    fn build(
        data: &dyn Array,
        _distance_type: lance_linalg::distance::DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self> {
        // Data-oblivious: we only need the dimension from the input data
        let dim = data.as_fixed_size_list().value_length() as usize;
        Self::new(dim, params)
    }

    fn retrain(&mut self, _data: &dyn Array) -> Result<()> {
        // Data-oblivious: no retraining needed
        Ok(())
    }

    fn code_dim(&self) -> usize {
        packed_len(self.metadata.dimension, self.metadata.num_bits)
    }

    fn column(&self) -> &'static str {
        TURBO_CODE_COLUMN
    }

    fn use_residual(distance_type: lance_linalg::distance::DistanceType) -> bool {
        matches!(
            distance_type,
            lance_linalg::distance::DistanceType::L2
                | lance_linalg::distance::DistanceType::Cosine
        )
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        let vectors = vectors.as_fixed_size_list();
        match vectors.value_type() {
            DataType::Float32 => self.quantize_f32(vectors),
            DataType::Float16 | DataType::Float64 => {
                // Convert to f32 first, then quantize
                // For now, only support f32 directly
                Err(Error::invalid_input(format!(
                    "TurboQuant currently only supports Float32, got {:?}. Cast to Float32 first.",
                    vectors.value_type()
                )))
            }
            dt => Err(Error::invalid_input(format!(
                "Unsupported data type for TurboQuant: {:?}",
                dt
            ))),
        }
    }

    fn metadata_key() -> &'static str {
        TURBO_METADATA_KEY
    }

    fn quantization_type() -> crate::vector::quantizer::QuantizationType {
        crate::vector::quantizer::QuantizationType::Turbo
    }

    fn metadata(&self, args: Option<QuantizationMetadata>) -> Self::Metadata {
        let mut metadata = self.metadata.clone();
        metadata.packed = args.map(|a| a.transposed).unwrap_or_default();
        metadata
    }

    fn from_metadata(
        metadata: &Self::Metadata,
        _distance_type: lance_linalg::distance::DistanceType,
    ) -> Result<Quantizer> {
        Ok(Quantizer::Turbo(Self {
            metadata: metadata.clone(),
        }))
    }

    fn field(&self) -> Field {
        Field::new(
            TURBO_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.code_dim() as i32,
            ),
            true,
        )
    }

    fn extra_fields(&self) -> Vec<Field> {
        vec![Field::new(TURBO_NORM_COLUMN, DataType::Float32, true)]
    }
}

impl TryFrom<Quantizer> for TurboQuantizer {
    type Error = Error;

    fn try_from(quantizer: Quantizer) -> Result<Self> {
        match quantizer {
            Quantizer::Turbo(q) => Ok(q),
            _ => Err(Error::invalid_input(
                "Cannot convert non-TurboQuantizer to TurboQuantizer",
            )),
        }
    }
}

impl From<TurboQuantizer> for Quantizer {
    fn from(quantizer: TurboQuantizer) -> Self {
        Self::Turbo(quantizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::turbo::packing::unpack_codes;
    use crate::vector::turbo::rotation::inverse_rotate;

    fn random_unit_vectors(n: usize, dim: usize) -> FixedSizeListArray {
        use rand::SeedableRng;
        use rand_distr::Distribution;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();

        let mut values = vec![0.0f32; n * dim];
        for i in 0..n {
            let start = i * dim;
            let end = start + dim;
            let slice = &mut values[start..end];
            for v in slice.iter_mut() {
                *v = normal.sample(&mut rng);
            }
            // Normalize to unit sphere
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            for v in slice.iter_mut() {
                *v /= norm;
            }
        }

        let float_array = Float32Array::from(values);
        FixedSizeListArray::try_new_from_values(float_array, dim as i32).unwrap()
    }

    #[test]
    fn test_quantize_shape() {
        let dim = 128;
        let params = TurboBuildParams {
            num_bits: 4,
            seed: 42,
        };
        let tq = TurboQuantizer::new(dim, &params).unwrap();

        let vectors = random_unit_vectors(10, dim);
        let codes = tq.quantize(&vectors).unwrap();
        let codes = codes.as_fixed_size_list();

        assert_eq!(codes.len(), 10);
        assert_eq!(codes.value_length() as usize, packed_len(dim, 4)); // 128*4/8 = 64
    }

    #[test]
    fn test_encode_decode_mse() {
        let dim = 256;
        let n_samples = 1000;

        for num_bits in [1, 2, 3, 4] {
            let params = TurboBuildParams {
                num_bits,
                seed: 42,
            };
            let tq = TurboQuantizer::new(dim, &params).unwrap();
            let (centroids, _boundaries) = tq.codebook().unwrap();
            let rotation = tq.rotation_flat();

            let vectors = random_unit_vectors(n_samples, dim);
            let values = vectors
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            let values_slice = values.values();

            let codes_array = tq.quantize(&vectors).unwrap();
            let codes_fsl = codes_array.as_fixed_size_list();
            let codes_values = codes_fsl
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap();
            let code_bytes = packed_len(dim, num_bits);

            let mut total_mse = 0.0f64;

            for i in 0..n_samples {
                let x = &values_slice[i * dim..(i + 1) * dim];
                let packed = &codes_values.values()[i * code_bytes..(i + 1) * code_bytes];

                // Unpack codes
                let indices = unpack_codes(packed, dim, num_bits).unwrap();

                // Dequantize: centroid lookup
                let y_hat: Vec<f32> = indices
                    .iter()
                    .map(|&idx| centroids[idx as usize])
                    .collect();

                // Inverse rotate
                let x_rec = inverse_rotate(&y_hat, &rotation, dim);

                // Compute MSE (unit vector, so no norm rescaling needed)
                let mse: f64 = x
                    .iter()
                    .zip(x_rec.iter())
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum();
                total_mse += mse;
            }

            let avg_mse = total_mse / n_samples as f64;

            // Compare against paper Theorem 1 bounds
            let bound = match num_bits {
                1 => 0.50,
                2 => 0.20,
                3 => 0.06,
                4 => 0.02,
                _ => 1.0,
            };
            assert!(
                avg_mse < bound,
                "b={}: avg MSE {} exceeds bound {}",
                num_bits,
                avg_mse,
                bound
            );
        }
    }

    #[test]
    fn test_invalid_params() {
        assert!(TurboQuantizer::new(2, &TurboBuildParams::default()).is_err());
        assert!(TurboQuantizer::new(
            128,
            &TurboBuildParams {
                num_bits: 0,
                seed: 42
            }
        )
        .is_err());
        assert!(TurboQuantizer::new(
            128,
            &TurboBuildParams {
                num_bits: 9,
                seed: 42
            }
        )
        .is_err());
    }
}
