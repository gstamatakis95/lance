"""Tests for rotation module."""

import numpy as np
import pytest
from scipy import stats

from turboquant.rotation import generate_rotation, inverse_rotate, rotate


class TestGenerateRotation:
    def test_orthogonality(self):
        d = 128
        R = generate_rotation(d, seed=42)
        identity = R @ R.T
        assert np.allclose(identity, np.eye(d), atol=1e-5)

    def test_shape(self):
        R = generate_rotation(256, seed=0)
        assert R.shape == (256, 256)
        assert R.dtype == np.float32

    def test_deterministic_with_seed(self):
        R1 = generate_rotation(64, seed=123)
        R2 = generate_rotation(64, seed=123)
        assert np.array_equal(R1, R2)

    def test_different_seeds_differ(self):
        R1 = generate_rotation(64, seed=1)
        R2 = generate_rotation(64, seed=2)
        assert not np.allclose(R1, R2)


class TestRotate:
    def test_norm_preservation_single(self):
        d = 512
        R = generate_rotation(d, seed=42)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(d).astype(np.float32)
        x /= np.linalg.norm(x)

        y = rotate(x, R)
        assert abs(np.linalg.norm(y) - 1.0) < 1e-5

    def test_norm_preservation_batch(self):
        d = 256
        n = 1000
        R = generate_rotation(d, seed=42)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d)).astype(np.float32)
        norms_before = np.linalg.norm(X, axis=1)

        Y = rotate(X, R)
        norms_after = np.linalg.norm(Y, axis=1)
        assert np.allclose(norms_before, norms_after, rtol=1e-4)

    def test_roundtrip(self):
        d = 128
        R = generate_rotation(d, seed=42)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(d).astype(np.float32)

        y = rotate(x, R)
        x_rec = inverse_rotate(y, R)
        assert np.allclose(x, x_rec, atol=1e-5)

    def test_roundtrip_batch(self):
        d = 128
        n = 100
        R = generate_rotation(d, seed=42)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d)).astype(np.float32)

        Y = rotate(X, R)
        X_rec = inverse_rotate(Y, R)
        assert np.allclose(X, X_rec, atol=1e-4)

    def test_coordinate_distribution_beta(self):
        """After rotation, coordinates of unit vectors ~ Beta on [-1, 1]."""
        d = 512
        n = 5000
        R = generate_rotation(d, seed=42)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        Y = rotate(X, R)
        # Check coordinate 0: should follow Beta((d-1)/2, (d-1)/2) on [-1,1]
        coord = Y[:, 0]
        # Transform to [0, 1] for Beta test
        coord_01 = (coord + 1) / 2
        alpha = (d - 1) / 2
        stat, p = stats.kstest(coord_01, stats.beta(alpha, alpha).cdf)
        assert p > 0.01, f"KS test failed: p={p:.4f}"
