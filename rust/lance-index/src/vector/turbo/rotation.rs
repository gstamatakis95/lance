// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Random rotation matrix generation for TurboQuant.
//!
//! Generates a Haar-distributed random orthogonal matrix Π ∈ R^{d×d} via QR
//! decomposition of a random Gaussian matrix (paper Section 3.1).
//!
//! # Algorithm
//!
//! 1. Generate d×d matrix G with i.i.d. N(0,1) entries (seeded RNG)
//! 2. Compute QR decomposition: G = Q·R
//! 3. Sign correction: `Q = Q · diag(sign(diag(R)))` to ensure uniqueness
//!
//! The resulting Q is Haar-distributed on the orthogonal group O(d).
//!
//! # Properties
//!
//! - **Deterministic**: Same seed always produces the same rotation matrix
//! - **Orthogonal**: Q^T · Q = I (preserves norms and angles)
//! - **Haar-distributed**: After rotation, coordinates are Beta-distributed (Lemma 1)
//!
//! # Storage
//!
//! For d=768: rotation matrix is 768×768 = 589,824 floats (~2.36 MB).
//! Stored as FixedSizeListArray in the index file's global buffer (protobuf Tensor).
//!
//! # Future: Hadamard rotation
//!
//! A randomized Hadamard transform (RHT) replaces the O(d²) dense matrix multiply
//! with O(d log d) via the Fast Walsh-Hadamard Transform. Storage drops from
//! 2.36 MB to just 96 bytes (768 sign bits). Not yet implemented.
//!
//! Reference: `design/turboquant-main/rotation.py`

use arrow_array::FixedSizeListArray;
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
use ndarray::s;
use num_traits::FromPrimitive;
use rand::SeedableRng;
use rand_distr::Distribution;

/// Generate a seeded random orthogonal matrix of dimension `dim × dim`.
///
/// Uses QR decomposition of a random Gaussian matrix with sign correction
/// to produce a Haar-distributed orthogonal matrix (same as paper's Π).
///
/// Returns as FixedSizeListArray with `dim` rows, each of `dim` f32 values.
pub fn generate_rotation_matrix<T: ArrowFloatType>(
    dim: usize,
    seed: u64,
) -> FixedSizeListArray
where
    T::Native: FromPrimitive,
{
    let mat = random_orthogonal_seeded::<T>(dim, seed);
    let (flat, _) = mat.into_raw_vec_and_offset();
    let values = <T::ArrayType as FloatArray<T>>::from_values(flat);
    FixedSizeListArray::try_new_from_values(values, dim as i32).unwrap()
}

/// Apply forward rotation: y = Π · x
///
/// For batch input of shape (n, dim), computes y[i] = Π · x[i] for each vector.
/// Rotation matrix is stored in row-major as a flat slice of length dim*dim.
///
/// In row-major convention: y_j = Σ_k rotation[j*dim + k] * x[k]
/// This is equivalent to: y = x @ Π^T in numpy row-vector convention.
pub fn rotate(vectors: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    let n = vectors.len() / dim;
    let mut output = vec![0.0f32; n * dim];

    for i in 0..n {
        let x = &vectors[i * dim..(i + 1) * dim];
        let y = &mut output[i * dim..(i + 1) * dim];

        for j in 0..dim {
            let mut sum = 0.0f32;
            let row = &rotation[j * dim..(j + 1) * dim];
            for k in 0..dim {
                sum += row[k] * x[k];
            }
            y[j] = sum;
        }
    }

    output
}

/// Apply inverse rotation: x̃ = Π^T · ỹ
///
/// Since Π is orthogonal, Π^(-1) = Π^T.
/// In row-major: x_j = Σ_k rotation[k*dim + j] * y[k] = Σ_k rotation^T[j*dim + k] * y[k]
/// This is equivalent to: x = y @ Π in numpy row-vector convention.
pub fn inverse_rotate(vectors: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    let n = vectors.len() / dim;
    let mut output = vec![0.0f32; n * dim];

    for i in 0..n {
        let y = &vectors[i * dim..(i + 1) * dim];
        let x = &mut output[i * dim..(i + 1) * dim];

        for j in 0..dim {
            let mut sum = 0.0f32;
            for k in 0..dim {
                // Π^T[j, k] = Π[k, j] = rotation[k*dim + j]
                sum += rotation[k * dim + j] * y[k];
            }
            x[j] = sum;
        }
    }

    output
}

/// Generate a d×d random Gaussian matrix with a specific seed.
fn random_normal_matrix_seeded(n: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    ndarray::Array2::from_shape_simple_fn((n, n), || normal.sample(&mut rng))
}

/// Householder QR decomposition.
///
/// Returns (Q, R) where Q is orthogonal and R is upper triangular.
/// Reference: https://en.wikipedia.org/wiki/Householder_transformation#QR_decomposition
fn householder_qr(a: ndarray::Array2<f64>) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    let (m, n) = a.dim();
    let mut q = ndarray::Array2::eye(m);
    let mut r = a;

    for k in 0..n.min(m - 1) {
        let mut x = r.slice(s![k.., k]).to_owned();
        let x_norm = x.dot(&x).sqrt();

        if x_norm < f64::EPSILON {
            continue;
        }

        // Create Householder vector
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * x_norm;
        let u = &x / x.dot(&x).sqrt();

        // Apply Householder transformation: H = I - 2*u*u^T
        let mut u_outer = ndarray::Array2::zeros((m - k, m - k));
        for i in 0..(m - k) {
            for j in 0..(m - k) {
                u_outer[[i, j]] = u[i] * u[j];
            }
        }
        let h = ndarray::Array2::eye(m - k) - 2.0 * u_outer;

        // Apply to R
        let r_block = r.slice(s![k.., k..]).to_owned();
        let h_r = h.dot(&r_block);
        r.slice_mut(s![k.., k..]).assign(&h_r);

        // Apply to Q
        let q_block = q.slice(s![.., k..]).to_owned();
        let q_h = q_block.dot(&h);
        q.slice_mut(s![.., k..]).assign(&q_h);
    }

    (q, r)
}

/// Generate a seeded random orthogonal matrix using QR decomposition.
///
/// The sign correction (multiplying Q columns by sign(diag(R))) ensures
/// the resulting matrix is Haar-distributed on the orthogonal group.
fn random_orthogonal_seeded<T: ArrowFloatType>(
    n: usize,
    seed: u64,
) -> ndarray::Array2<T::Native>
where
    T::Native: FromPrimitive,
{
    let a = random_normal_matrix_seeded(n, seed);
    let (mut q, r) = householder_qr(a);

    // Sign correction: Q = Q * diag(sign(diag(R)))
    // This ensures the decomposition is unique and Haar-distributed
    for j in 0..n {
        let sign = if r[[j, j]] >= 0.0 { 1.0 } else { -1.0 };
        let mut col = q.column_mut(j);
        col.mapv_inplace(|v| v * sign);
    }

    q.mapv(|v| T::Native::from_f64(v).unwrap())
}

/// Extract the rotation matrix as a flat f32 slice from a FixedSizeListArray.
pub fn rotation_as_flat_f32(rotation: &FixedSizeListArray) -> Vec<f32> {
    let values = rotation.values();
    values
        .as_any()
        .downcast_ref::<arrow_array::Float32Array>()
        .unwrap()
        .values()
        .to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Float32Type;
    use arrow_array::Array;

    const TEST_DIM: usize = 32;
    const TEST_SEED: u64 = 42;

    fn get_test_rotation() -> Vec<f32> {
        let mat = generate_rotation_matrix::<Float32Type>(TEST_DIM, TEST_SEED);
        rotation_as_flat_f32(&mat)
    }

    #[test]
    fn test_orthogonality() {
        let r = get_test_rotation();
        // R @ R^T should be identity
        for i in 0..TEST_DIM {
            for j in 0..TEST_DIM {
                let mut dot = 0.0f32;
                for k in 0..TEST_DIM {
                    dot += r[i * TEST_DIM + k] * r[j * TEST_DIM + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "R@R^T[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_deterministic_with_seed() {
        let r1 = get_test_rotation();
        let r2 = get_test_rotation();
        assert_eq!(r1, r2, "Same seed should produce identical rotation");
    }

    #[test]
    fn test_different_seeds_differ() {
        let r1 = generate_rotation_matrix::<Float32Type>(TEST_DIM, 42);
        let r2 = generate_rotation_matrix::<Float32Type>(TEST_DIM, 43);
        let flat1 = rotation_as_flat_f32(&r1);
        let flat2 = rotation_as_flat_f32(&r2);
        assert_ne!(flat1, flat2, "Different seeds should produce different rotations");
    }

    #[test]
    fn test_norm_preservation() {
        let r = get_test_rotation();
        // Random unit vector
        let mut x = vec![0.0f32; TEST_DIM];
        x[0] = 1.0;
        x[1] = 0.5;
        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        let y = rotate(&x, &r, TEST_DIM);
        let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            (norm_x - norm_y).abs() < 1e-4,
            "Norm not preserved: {} vs {}",
            norm_x,
            norm_y
        );
    }

    #[test]
    fn test_roundtrip() {
        let r = get_test_rotation();
        let x: Vec<f32> = (0..TEST_DIM).map(|i| (i as f32 + 1.0) / TEST_DIM as f32).collect();

        let y = rotate(&x, &r, TEST_DIM);
        let x_rec = inverse_rotate(&y, &r, TEST_DIM);

        for i in 0..TEST_DIM {
            assert!(
                (x[i] - x_rec[i]).abs() < 1e-4,
                "Roundtrip failed at index {}: {} vs {}",
                i,
                x[i],
                x_rec[i]
            );
        }
    }

    #[test]
    fn test_batch_roundtrip() {
        let r = get_test_rotation();
        let n = 10;
        let x: Vec<f32> = (0..n * TEST_DIM)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();

        let y = rotate(&x, &r, TEST_DIM);
        let x_rec = inverse_rotate(&y, &r, TEST_DIM);

        for i in 0..x.len() {
            assert!(
                (x[i] - x_rec[i]).abs() < 1e-3,
                "Batch roundtrip failed at index {}: {} vs {}",
                i,
                x[i],
                x_rec[i]
            );
        }
    }

    #[test]
    fn test_shape() {
        let mat = generate_rotation_matrix::<Float32Type>(TEST_DIM, TEST_SEED);
        assert_eq!(mat.len(), TEST_DIM);
        assert_eq!(mat.value_length(), TEST_DIM as i32);
    }
}
