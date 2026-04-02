"""Tests for core TurboQuant quantizers."""

import numpy as np
import pytest

from turboquant.core import TurboQuantMSE, TurboQuantProd
from turboquant.utils import random_unit_vectors


class TestTurboQuantMSE:
    def test_roundtrip_shape_single(self):
        d, b = 128, 2
        tq = TurboQuantMSE(d, b, seed=42)
        x = random_unit_vectors(1, d, seed=0)[0]
        q = tq.quantize(x)
        x_rec = tq.dequantize(q)
        assert x_rec.shape == (d,)

    def test_roundtrip_shape_batch(self):
        d, b, n = 128, 2, 50
        tq = TurboQuantMSE(d, b, seed=42)
        X = random_unit_vectors(n, d, seed=0)
        q = tq.quantize(X)
        X_rec = tq.dequantize(q)
        assert X_rec.shape == (n, d)

    def test_non_unit_vectors(self):
        """Should handle vectors with arbitrary norms."""
        d, b = 128, 3
        tq = TurboQuantMSE(d, b, seed=42)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(d).astype(np.float32) * 5.0
        q = tq.quantize(x)
        x_rec = tq.dequantize(q)
        # Norm should be approximately preserved
        assert abs(np.linalg.norm(x_rec) - np.linalg.norm(x)) / np.linalg.norm(x) < 0.2

    @pytest.mark.parametrize("b,max_distortion", [
        (1, 0.50),
        (2, 0.18),
        (3, 0.05),
        (4, 0.015),
    ])
    def test_distortion_matches_paper(self, b, max_distortion):
        """Empirical MSE distortion should be within tolerance of paper values."""
        d = 256
        tq = TurboQuantMSE(d, b, seed=42)
        dist = tq.distortion(n_samples=5000, seed=0)
        assert dist < max_distortion, f"b={b}: distortion {dist:.4f} > {max_distortion}"

    def test_higher_b_lower_distortion(self):
        d = 256
        distortions = []
        for b in [1, 2, 3, 4]:
            tq = TurboQuantMSE(d, b, seed=42)
            distortions.append(tq.distortion(n_samples=3000))
        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i + 1]

    def test_invalid_b(self):
        with pytest.raises(ValueError):
            TurboQuantMSE(128, 0)
        with pytest.raises(ValueError):
            TurboQuantMSE(128, 9)

    def test_invalid_d(self):
        with pytest.raises(ValueError):
            TurboQuantMSE(2, 2)


class TestTurboQuantProd:
    def test_invalid_b(self):
        with pytest.raises(ValueError):
            TurboQuantProd(128, 1)  # b must be >= 2

    def test_roundtrip_shape(self):
        d, b = 128, 3
        tq = TurboQuantProd(d, b, seed=42)
        x = random_unit_vectors(1, d, seed=0)[0]
        q = tq.quantize(x)
        x_rec = tq.dequantize(q)
        assert x_rec.shape == (d,)

    def test_unbiased_inner_product(self):
        """Mean inner product estimation error should be near zero."""
        d, b = 256, 3
        n = 3000
        tq = TurboQuantProd(d, b, seed=42)
        X = random_unit_vectors(n, d, seed=0)
        Y = random_unit_vectors(n, d, seed=1)

        true_ips = np.sum(X * Y, axis=1)
        errors = []
        for i in range(n):
            q = tq.quantize(X[i])
            est = tq.inner_product(q, Y[i])
            errors.append(float(est) - true_ips[i])

        bias = np.mean(errors)
        assert abs(bias) < 0.02, f"Bias too large: {bias:.4f}"

    def test_distortion_ip(self):
        d, b = 128, 3
        tq = TurboQuantProd(d, b, seed=42)
        result = tq.distortion_ip(n_samples=2000, seed=0)
        assert abs(result["bias"]) < 0.03
        assert result["variance"] > 0
        assert result["mse"] > 0

    def test_inner_product_method_matches_dequantize(self):
        """inner_product() should give same result as dequantize() + dot."""
        d, b = 128, 3
        tq = TurboQuantProd(d, b, seed=42)
        x = random_unit_vectors(1, d, seed=0)[0]
        y = random_unit_vectors(1, d, seed=1)[0]

        q = tq.quantize(x)
        ip_method = float(tq.inner_product(q, y))
        ip_manual = float(np.dot(tq.dequantize(q), y))
        assert abs(ip_method - ip_manual) < 1e-4

    def test_higher_b_lower_variance(self):
        """More bits should reduce inner product error variance."""
        d = 128
        variances = []
        for b in [2, 3, 4]:
            tq = TurboQuantProd(d, b, seed=42)
            result = tq.distortion_ip(n_samples=2000, seed=0)
            variances.append(result["variance"])
        for i in range(len(variances) - 1):
            assert variances[i] > variances[i + 1]
