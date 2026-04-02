"""Tests for QJL module."""

import numpy as np
import pytest

from turboquant.qjl import generate_projection, qjl_dequantize, qjl_inner_product, qjl_quantize
from turboquant.utils import random_unit_vectors


class TestQJLBasics:
    def test_output_shape_single(self):
        d = 128
        S = generate_projection(d, seed=42)
        x = random_unit_vectors(1, d, seed=0)[0]
        z = qjl_quantize(x, S)
        assert z.shape == (d,)
        assert z.dtype == np.int8

    def test_output_values_are_signs(self):
        d = 256
        S = generate_projection(d, seed=42)
        X = random_unit_vectors(100, d, seed=0)
        Z = qjl_quantize(X, S)
        assert set(np.unique(Z)).issubset({-1, 1})

    def test_output_shape_batch(self):
        d = 128
        n = 50
        S = generate_projection(d, seed=42)
        X = random_unit_vectors(n, d, seed=0)
        Z = qjl_quantize(X, S)
        assert Z.shape == (n, d)

    def test_deterministic(self):
        d = 64
        S = generate_projection(d, seed=42)
        x = random_unit_vectors(1, d, seed=0)[0]
        z1 = qjl_quantize(x, S)
        z2 = qjl_quantize(x, S)
        assert np.array_equal(z1, z2)


class TestQJLUnbiasedness:
    def test_unbiased_inner_product(self):
        """Over many random pairs, mean estimation error should be near zero."""
        d = 256
        n = 5000
        S = generate_projection(d, seed=42)
        X = random_unit_vectors(n, d, seed=0)
        Y = random_unit_vectors(n, d, seed=1)

        true_ips = np.sum(X * Y, axis=1)
        errors = []
        for i in range(n):
            z = qjl_quantize(X[i], S)
            res_norm = np.float32(np.linalg.norm(X[i]))  # = 1.0 for unit vectors
            est_ip = qjl_inner_product(z, Y[i], S, res_norm)
            errors.append(float(est_ip) - true_ips[i])

        mean_error = np.mean(errors)
        assert abs(mean_error) < 0.02, f"QJL bias too large: {mean_error:.4f}"

    def test_variance_scales_with_d(self):
        """Variance should decrease roughly as 1/d."""
        variances = {}
        for d in [128, 512]:
            n = 3000
            S = generate_projection(d, seed=42)
            X = random_unit_vectors(n, d, seed=0)
            Y = random_unit_vectors(n, d, seed=1)
            true_ips = np.sum(X * Y, axis=1)

            errors = []
            for i in range(n):
                z = qjl_quantize(X[i], S)
                est_ip = qjl_inner_product(z, Y[i], S, np.float32(1.0))
                errors.append(float(est_ip) - true_ips[i])
            variances[d] = np.var(errors)

        # Doubling d should roughly halve variance
        ratio = variances[128] / variances[512]
        assert ratio > 2.0, f"Variance ratio {ratio:.2f} should be > 2"
