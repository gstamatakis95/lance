"""TurboQuant core quantizers — MSE-optimal and inner-product-optimal.

TurboQuantMSE: For Value cache / vector reconstruction (minimizes ‖x - x̃‖²).
TurboQuantProd: For Key cache / inner product preservation (unbiased ⟨y, x̃⟩ = ⟨y, x⟩).
"""

from dataclasses import dataclass

import numpy as np

from .qjl import generate_projection, qjl_dequantize, qjl_inner_product, qjl_quantize
from .rotation import generate_rotation, inverse_rotate, rotate
from .scalar_quantizer import compute_centroids, dequantize_scalar, quantize_scalar
from .utils import normalize


@dataclass(frozen=True)
class QuantizedMSE:
    """Compressed representation from TurboQuantMSE."""
    indices: np.ndarray   # uint8, shape (d,) or (n, d)
    norms: np.ndarray     # float32, scalar or shape (n,)


@dataclass(frozen=True)
class QuantizedProd:
    """Compressed representation from TurboQuantProd."""
    mse_indices: np.ndarray    # uint8, shape (d,) or (n, d)
    qjl_signs: np.ndarray      # int8, shape (d,) or (n, d)
    residual_norm: np.ndarray  # float32, scalar or shape (n,)
    input_norm: np.ndarray     # float32, scalar or shape (n,)


class TurboQuantMSE:
    """MSE-optimal quantizer via random rotation + Beta-distribution scalar quantization.

    Args:
        d: Vector dimension.
        b: Bit-width per coordinate (1-8).
        seed: Random seed for reproducibility.
    """

    def __init__(self, d: int, b: int, seed: int | None = None):
        if b < 1 or b > 8:
            raise ValueError(f"Bit-width b must be in [1, 8], got {b}")
        if d < 3:
            raise ValueError(f"Dimension d must be >= 3, got {d}")
        self.d = d
        self.b = b
        self.rotation = generate_rotation(d, seed)
        centroids_boundaries = compute_centroids(d, b)
        self.centroids = centroids_boundaries[0]
        self.boundaries = centroids_boundaries[1]

    def quantize(self, x: np.ndarray) -> QuantizedMSE:
        """Quantize vector(s) to compressed representation.

        Args:
            x: Input vector(s), shape (d,) or (n, d). Need not be unit norm.

        Returns:
            QuantizedMSE with bin indices and original norms.
        """
        x = np.asarray(x, dtype=np.float32)
        x_hat, norms = normalize(x)
        y = rotate(x_hat, self.rotation)
        indices = quantize_scalar(y, self.boundaries)
        return QuantizedMSE(indices=indices, norms=norms)

    def dequantize(self, q: QuantizedMSE) -> np.ndarray:
        """Reconstruct vector(s) from compressed representation.

        Args:
            q: QuantizedMSE from quantize().

        Returns:
            Reconstructed vector(s), same shape as original input.
        """
        y_hat = dequantize_scalar(q.indices, self.centroids)
        x_hat = inverse_rotate(y_hat, self.rotation)
        if q.norms.ndim == 0:
            return x_hat * float(q.norms)
        else:
            return x_hat * q.norms[:, np.newaxis]

    def distortion(self, n_samples: int = 10000, seed: int | None = 42) -> float:
        """Empirical MSE distortion over random unit vectors.

        Returns:
            Mean ‖x - dequant(quant(x))‖²₂ over n_samples.
        """
        from .utils import random_unit_vectors
        X = random_unit_vectors(n_samples, self.d, seed)
        q = self.quantize(X)
        X_hat = self.dequantize(q)
        mse = np.mean(np.sum((X - X_hat) ** 2, axis=1))
        return float(mse)


class TurboQuantProd:
    """Inner-product-optimal quantizer: MSE quantization + QJL residual correction.

    Provides unbiased inner product estimation: 𝔼[⟨y, x̃⟩] = ⟨y, x⟩.
    Allocates (b-1) bits to MSE quantizer and 1 bit to QJL on the residual.

    Args:
        d: Vector dimension.
        b: Total bit-width per coordinate (≥ 2).
        seed: Random seed for reproducibility.
    """

    def __init__(self, d: int, b: int, seed: int | None = None):
        if b < 2:
            raise ValueError(f"TurboQuantProd requires b >= 2, got {b}")
        if d < 3:
            raise ValueError(f"Dimension d must be >= 3, got {d}")
        self.d = d
        self.b = b
        # b-1 bits for MSE, 1 bit for QJL
        self.mse_quantizer = TurboQuantMSE(d, b - 1, seed)
        # QJL projection matrix (different seed to avoid correlation)
        qjl_seed = seed + 1000 if seed is not None else None
        self.S = generate_projection(d, qjl_seed)

    def quantize(self, x: np.ndarray) -> QuantizedProd:
        """Quantize vector(s) with MSE + QJL residual correction.

        Args:
            x: Input vector(s), shape (d,) or (n, d).

        Returns:
            QuantizedProd with MSE indices, QJL signs, and norms.
        """
        x = np.asarray(x, dtype=np.float32)
        x_hat, input_norms = normalize(x)

        # MSE quantize the unit vector
        q_mse = self.mse_quantizer.quantize(x_hat)
        x_mse = self.mse_quantizer.dequantize(q_mse)

        # Compute residual
        residual = x_hat - x_mse
        if residual.ndim == 1:
            res_norm = np.float32(np.linalg.norm(residual))
        else:
            res_norm = np.linalg.norm(residual, axis=1).astype(np.float32)

        # QJL quantize the residual
        signs = qjl_quantize(residual, self.S)

        return QuantizedProd(
            mse_indices=q_mse.indices,
            qjl_signs=signs,
            residual_norm=res_norm,
            input_norm=input_norms,
        )

    def dequantize(self, q: QuantizedProd) -> np.ndarray:
        """Reconstruct vector(s). Unbiased for inner products, biased for MSE.

        Args:
            q: QuantizedProd from quantize().

        Returns:
            Reconstructed vector(s).
        """
        # MSE reconstruction (unit sphere)
        q_mse = QuantizedMSE(indices=q.mse_indices, norms=np.float32(1.0))
        x_mse = self.mse_quantizer.dequantize(q_mse)

        # QJL reconstruction of residual
        x_qjl = qjl_dequantize(q.qjl_signs, self.S, q.residual_norm)

        x_hat = x_mse + x_qjl

        # Rescale by original norm
        if q.input_norm.ndim == 0:
            return x_hat * float(q.input_norm)
        else:
            return x_hat * q.input_norm[:, np.newaxis]

    def inner_product(self, q: QuantizedProd, y: np.ndarray) -> np.ndarray:
        """Compute ⟨y, x⟩ from quantized x without full vector reconstruction.

        More efficient than dequantize() followed by dot product.

        Args:
            q: QuantizedProd from quantize().
            y: Query vector(s), shape (d,) or (m, d).

        Returns:
            Estimated inner product(s). Unbiased: 𝔼[result] = ⟨y, x⟩.
        """
        # MSE component
        q_mse = QuantizedMSE(indices=q.mse_indices, norms=np.float32(1.0))
        x_mse = self.mse_quantizer.dequantize(q_mse)

        if x_mse.ndim == 1 and y.ndim == 1:
            ip_mse = np.dot(y, x_mse)
        elif x_mse.ndim == 2 and y.ndim == 1:
            ip_mse = x_mse @ y
        else:
            ip_mse = x_mse @ y.T

        # QJL component
        ip_qjl = qjl_inner_product(q.qjl_signs, y, self.S, q.residual_norm)

        result = (ip_mse + ip_qjl)

        # Scale by input norm
        if q.input_norm.ndim == 0:
            return result * float(q.input_norm)
        else:
            if result.ndim == 1:
                return result * q.input_norm
            else:
                return result * q.input_norm[:, np.newaxis]

    def distortion_ip(self, n_samples: int = 10000, seed: int | None = 42) -> dict:
        """Empirical inner product distortion statistics.

        Returns:
            Dict with 'bias' (mean error), 'variance' (error variance),
            'mse' (mean squared error of inner product).
        """
        from .utils import random_unit_vectors
        rng = np.random.default_rng(seed)

        X = random_unit_vectors(n_samples, self.d, seed)
        Y = random_unit_vectors(n_samples, self.d, seed + 1 if seed else 1)

        true_ip = np.sum(X * Y, axis=1)  # (n,)

        # Quantize X, compute inner product with Y
        errors = []
        for i in range(n_samples):
            q = self.quantize(X[i])
            est_ip = self.inner_product(q, Y[i])
            errors.append(float(est_ip) - true_ip[i])

        errors = np.array(errors)
        return {
            "bias": float(np.mean(errors)),
            "variance": float(np.var(errors)),
            "mse": float(np.mean(errors ** 2)),
        }
