"""Tests for scalar quantizer module."""

import numpy as np
import pytest

from turboquant.scalar_quantizer import (
    compute_centroids,
    dequantize_scalar,
    per_coordinate_distortion,
    quantize_scalar,
)


class TestComputeCentroids:
    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_correct_count(self, b):
        centroids, boundaries = compute_centroids(128, b)
        assert len(centroids) == 2 ** b
        assert len(boundaries) == 2 ** b + 1

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_symmetry(self, b):
        """Centroids should be symmetric around 0."""
        centroids, _ = compute_centroids(512, b)
        k = len(centroids)
        for i in range(k // 2):
            assert abs(centroids[i] + centroids[k - 1 - i]) < 1e-6

    def test_boundaries_ordered(self):
        _, boundaries = compute_centroids(256, 3)
        assert all(boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1))

    def test_boundaries_endpoints(self):
        _, boundaries = compute_centroids(256, 2)
        assert boundaries[0] == -1.0
        assert boundaries[-1] == 1.0

    def test_centroids_within_range(self):
        centroids, _ = compute_centroids(256, 4)
        assert np.all(centroids >= -1.0)
        assert np.all(centroids <= 1.0)

    def test_caching(self):
        """Same (d, b) should return identical results."""
        c1, b1 = compute_centroids(128, 2)
        c2, b2 = compute_centroids(128, 2)
        assert np.array_equal(c1, c2)
        assert np.array_equal(b1, b2)


class TestQuantizeDequantize:
    def test_roundtrip_indices(self):
        centroids, boundaries = compute_centroids(256, 2)
        values = centroids.copy()  # Quantizing centroids should return themselves
        indices = quantize_scalar(values, boundaries)
        reconstructed = dequantize_scalar(indices, centroids)
        assert np.allclose(values, reconstructed, atol=1e-6)

    def test_output_dtype(self):
        _, boundaries = compute_centroids(256, 3)
        values = np.array([0.0, 0.1, -0.1], dtype=np.float32)
        indices = quantize_scalar(values, boundaries)
        assert indices.dtype == np.uint8

    def test_batch_shape(self):
        _, boundaries = compute_centroids(128, 2)
        values = np.random.randn(100, 128).astype(np.float32) * 0.1
        indices = quantize_scalar(values, boundaries)
        assert indices.shape == (100, 128)


class TestDistortion:
    @pytest.mark.parametrize("b,expected_max", [
        (1, 0.50),   # paper: 0.36, allow generous margin
        (2, 0.18),   # paper: 0.117
        (3, 0.05),   # paper: 0.03
        (4, 0.015),  # paper: 0.009
    ])
    def test_per_coordinate_matches_paper(self, b, expected_max):
        """Per-coordinate distortion × d should match paper's total MSE distortion."""
        d = 512  # Use moderate d for speed
        pcd = per_coordinate_distortion(d, b)
        total = d * pcd
        assert total < expected_max, f"b={b}: total distortion {total:.4f} > {expected_max}"
