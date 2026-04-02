// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! TurboQuant: data-oblivious vector quantization with near-optimal distortion.
//!
//! Implements the TurboQuantMSE algorithm from Zandieh et al. (ICLR 2026,
//! arXiv:2504.19874). TurboQuant compresses vectors by:
//!
//! 1. **Rotating** with a random orthogonal matrix Π, inducing Beta-distributed
//!    coordinates (paper Lemma 1).
//! 2. **Quantizing** each coordinate independently using a precomputed optimal
//!    Lloyd-Max codebook for the Beta distribution (paper Eq. 4).
//! 3. **Packing** the b-bit indices into a compact byte array.
//!
//! # Key Properties
//!
//! - **Data-oblivious**: Codebooks depend only on `(dimension, bit_width)`, not on
//!   any data. Training takes <1ms vs minutes for PQ.
//! - **Near-optimal**: MSE distortion is within 2.7x of the information-theoretic
//!   lower bound (paper Theorem 1).
//! - **Parallelizable**: Deterministic training enables trivially parallel
//!   distributed builds at billion scale.
//!
//! # Compression (d=768)
//!
//! | Bits | Bytes/vector | Compression vs fp32 | Recall@10 |
//! |------|-------------|---------------------|-----------|
//! | 1    | 100         | 31x                 | ~70%      |
//! | 2    | 196         | 16x                 | ~85%      |
//! | 4    | 388         | 8x                  | ~95%      |
//! | 8    | 772         | 4x                  | ~99%      |
//!
//! # Module Structure
//!
//! - [`codebook`]: Lloyd-Max optimal codebook for Beta distribution
//! - [`rotation`]: Random orthogonal matrix via QR decomposition
//! - [`packing`]: b-bit pack/unpack utilities
//! - [`builder`]: [`TurboQuantizer`](builder::TurboQuantizer) implementing the
//!   [`Quantization`](crate::vector::quantizer::Quantization) trait
//! - [`storage`]: On-disk storage, metadata serialization, asymmetric distance calculation
//! - [`transform`]: IVF transform pipeline integration
//!
//! # Usage
//!
//! ```ignore
//! use lance_index::vector::turbo::{TurboBuildParams, builder::TurboQuantizer};
//!
//! let params = TurboBuildParams { num_bits: 4, seed: 42 };
//! let tq = TurboQuantizer::new(768, &params)?;
//! let codes = tq.quantize(&vectors)?;
//! ```

pub mod builder;
pub mod codebook;
pub mod packing;
pub mod rotation;
pub mod storage;
pub mod transform;

use crate::vector::quantizer::QuantizerBuildParams;
use lance_linalg::distance::DistanceType;

/// Build parameters for TurboQuantizer.
///
/// Unlike PQ/SQ, TurboQuant doesn't need to sample any training data —
/// the codebook is determined solely by (dimension, num_bits).
#[derive(Debug, Clone)]
pub struct TurboBuildParams {
    /// Bit-width per coordinate (1-8). Default: 4.
    pub num_bits: u32,
    /// RNG seed for rotation matrix generation. Default: 42.
    pub seed: u64,
}

impl Default for TurboBuildParams {
    fn default() -> Self {
        Self {
            num_bits: 4,
            seed: 42,
        }
    }
}

impl QuantizerBuildParams for TurboBuildParams {
    fn sample_size(&self) -> usize {
        // TurboQuant is data-oblivious — no sampling needed
        0
    }

    fn use_residual(distance_type: DistanceType) -> bool {
        matches!(distance_type, DistanceType::L2 | DistanceType::Cosine)
    }
}
