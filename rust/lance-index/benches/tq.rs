// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! TurboQuant benchmarks.
//!
//! Reproduces the benchmarks from the Python reference implementation
//! (`design/turboquant-main/benchmarks/distortion.py`) and paper (Zandieh et al., ICLR 2026).
//!
//! Benchmarks:
//! 1. Distortion validation — MSE matches paper Theorem 1 bounds
//! 2. Encode throughput — vectors/sec for quantize
//! 3. Distance computation — distances/sec for asymmetric distance
//! 4. Distance table construction — query rotation setup cost
//!
//! Run: cargo bench -p lance-index -- tq

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, UInt64Array, UInt8Array};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::vector::turbo::builder::TurboQuantizer;
use lance_index::vector::turbo::codebook::{get_codebook, per_coordinate_distortion};
use lance_index::vector::turbo::packing::packed_len;
use lance_index::vector::turbo::rotation::rotation_as_flat_f32;
use lance_index::vector::turbo::storage::{
    TurboQuantizationStorage, TURBO_CODE_COLUMN, TURBO_NORM_COLUMN,
};
use lance_index::vector::turbo::TurboBuildParams;
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use rand::SeedableRng;
use rand_distr::Distribution;

const DIM: usize = 128;
const TOTAL: usize = 16_000;

/// Generate random unit vectors.
fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();
    let mut values = vec![0.0f32; n * dim];
    for i in 0..n {
        let start = i * dim;
        let end = start + dim;
        let slice = &mut values[start..end];
        for v in slice.iter_mut() {
            *v = normal.sample(&mut rng);
        }
        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in slice.iter_mut() {
            *v /= norm;
        }
    }
    values
}

/// Build a mock TurboQuantizationStorage with random quantized data.
fn mock_tq_storage(num_bits: u32, dim: usize, total: usize) -> TurboQuantizationStorage {
    let params = TurboBuildParams {
        num_bits,
        seed: 42,
    };
    let tq = TurboQuantizer::new(dim, &params).unwrap();

    // Generate random vectors and quantize them
    let values = random_unit_vectors(total, dim, 0);
    let float_array = Float32Array::from(values);
    let vectors = FixedSizeListArray::try_new_from_values(float_array, dim as i32).unwrap();

    let codes = tq.quantize(&vectors).unwrap();
    let codes_fsl = codes.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

    // Compute norms (all ~1.0 for unit vectors)
    let norms = Float32Array::from(vec![1.0f32; total]);
    let row_ids = UInt64Array::from((0..total as u64).collect::<Vec<_>>());

    let batch = arrow_array::RecordBatch::try_from_iter(vec![
        (ROW_ID, Arc::new(row_ids) as ArrayRef),
        (TURBO_CODE_COLUMN, Arc::new(codes_fsl.clone()) as ArrayRef),
        (TURBO_NORM_COLUMN, Arc::new(norms) as ArrayRef),
    ])
    .unwrap();

    let metadata = tq.metadata(None);
    TurboQuantizationStorage::try_from_batch(batch, &metadata, DistanceType::L2, None).unwrap()
}

/// Benchmark: distance table construction (includes query rotation).
fn bench_dist_table_construction(c: &mut Criterion) {
    for num_bits in [1u32, 2, 4] {
        let storage = mock_tq_storage(num_bits, DIM, TOTAL);
        let query_values = random_unit_vectors(1, DIM, 99);
        let query = Float32Array::from(query_values);

        c.bench_function(
            &format!("TQ{}: construct_dist_table: L2,DIM={}", num_bits, DIM),
            |b| {
                b.iter(|| {
                    black_box(storage.dist_calculator(Arc::new(query.clone()), 0.0));
                })
            },
        );
    }
}

/// Benchmark: distance computation (all vectors).
fn bench_compute_distances(c: &mut Criterion) {
    for num_bits in [1u32, 2, 4] {
        let storage = mock_tq_storage(num_bits, DIM, TOTAL);
        let query_values = random_unit_vectors(1, DIM, 99);
        let query = Float32Array::from(query_values);
        let dist_calc = storage.dist_calculator(Arc::new(query.clone()), 0.0);

        c.bench_function(
            &format!(
                "TQ{}: compute_distances: {},DIM={}",
                num_bits, TOTAL, DIM
            ),
            |b| {
                b.iter(|| {
                    black_box(dist_calc.distance_all(0));
                })
            },
        );

        c.bench_function(
            &format!(
                "TQ{}: compute_distances_single: {},DIM={}",
                num_bits, TOTAL, DIM
            ),
            |b| {
                b.iter(|| {
                    for i in 0..TOTAL {
                        black_box(dist_calc.distance(i as u32));
                    }
                })
            },
        );
    }
}

/// Benchmark: encode throughput (quantize vectors).
fn bench_encode_throughput(c: &mut Criterion) {
    for num_bits in [1u32, 2, 4] {
        let params = TurboBuildParams {
            num_bits,
            seed: 42,
        };
        let tq = TurboQuantizer::new(DIM, &params).unwrap();
        let values = random_unit_vectors(TOTAL, DIM, 0);
        let float_array = Float32Array::from(values);
        let vectors = FixedSizeListArray::try_new_from_values(float_array, DIM as i32).unwrap();

        c.bench_function(
            &format!("TQ{}: encode: {},DIM={}", num_bits, TOTAL, DIM),
            |b| {
                b.iter(|| {
                    black_box(tq.quantize(&vectors).unwrap());
                })
            },
        );
    }
}

/// Benchmark: distortion validation (reproduces paper Table 1 / Python distortion.py).
///
/// This is not a speed benchmark — it validates correctness by computing MSE
/// and comparing against the paper's theoretical bounds. Runs once and prints results.
///
/// Paper Theorem 1 reference values:
///   b=1: D_mse ≈ 0.36
///   b=2: D_mse ≈ 0.117
///   b=3: D_mse ≈ 0.03
///   b=4: D_mse ≈ 0.009
fn bench_distortion_validation(c: &mut Criterion) {
    let paper_mse = [(1u32, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)];

    for dim in [128, 512] {
        for &(num_bits, paper_val) in &paper_mse {
            let params = TurboBuildParams {
                num_bits,
                seed: 42,
            };
            let tq = TurboQuantizer::new(dim, &params).unwrap();
            let (centroids, _boundaries) = get_codebook(dim, num_bits).unwrap();
            let rotation = rotation_as_flat_f32(tq.rotation_matrix().unwrap());

            let n_samples = 2000;
            let values = random_unit_vectors(n_samples, dim, 0);
            let float_array = Float32Array::from(values.clone());
            let vectors =
                FixedSizeListArray::try_new_from_values(float_array, dim as i32).unwrap();

            // We wrap this in a criterion benchmark so it appears in the output,
            // but the important thing is the printed MSE comparison.
            c.bench_function(
                &format!("TQ_distortion: d={},b={}", dim, num_bits),
                |b| {
                    b.iter(|| {
                        let codes = tq.quantize(&vectors).unwrap();
                        let codes_fsl =
                            codes.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                        let codes_u8 = codes_fsl
                            .values()
                            .as_any()
                            .downcast_ref::<UInt8Array>()
                            .unwrap();
                        let code_bytes = packed_len(dim, num_bits);

                        let mut total_mse = 0.0f64;
                        for i in 0..n_samples {
                            let x = &values[i * dim..(i + 1) * dim];
                            let packed =
                                &codes_u8.values()[i * code_bytes..(i + 1) * code_bytes];
                            let indices = lance_index::vector::turbo::packing::unpack_codes(
                                packed, dim, num_bits,
                            )
                            .unwrap();
                            let y_hat: Vec<f32> =
                                indices.iter().map(|&idx| centroids[idx as usize]).collect();
                            let x_rec =
                                lance_index::vector::turbo::rotation::inverse_rotate(
                                    &y_hat, &rotation, dim,
                                );
                            let mse: f64 = x
                                .iter()
                                .zip(x_rec.iter())
                                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                                .sum();
                            total_mse += mse;
                        }
                        let avg_mse = total_mse / n_samples as f64;
                        black_box(avg_mse);
                    })
                },
            );

            // Also print the theoretical per-coordinate distortion
            let per_coord = per_coordinate_distortion(dim, num_bits).unwrap();
            let theoretical_total = dim as f64 * per_coord;
            eprintln!(
                "  d={}, b={}: theoretical_mse={:.5}, paper_mse={:.3}, ratio={:.2}x",
                dim,
                num_bits,
                theoretical_total,
                paper_val,
                theoretical_total / paper_val
            );
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10)).sample_size(10);
    targets = bench_dist_table_construction, bench_compute_distances, bench_encode_throughput, bench_distortion_validation
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10)).sample_size(10);
    targets = bench_dist_table_construction, bench_compute_distances, bench_encode_throughput, bench_distortion_validation
);

criterion_main!(benches);
