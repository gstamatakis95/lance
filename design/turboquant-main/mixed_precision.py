"""Mixed Precision Quantizer — Outlier-aware channel-level bit allocation.

Key insight: Modern LLMs (Qwen, Llama) have K vectors where ~5-20% of channels
have RMS values 10-100x larger than the median. These outlier channels dominate
quantization error. Solution: store outlier channels at FP16, quantize the rest
at low bit-width. Average bit-width ≈ outlier_ratio × 16 + (1 - outlier_ratio) × low_bits.

Example: 10% outlier at FP16 + 90% at 3-bit → average 4.3 bits.
"""

from dataclasses import dataclass

import numpy as np

from .core import TurboQuantMSE


@dataclass(frozen=True)
class QuantizedMixed:
    """Compressed representation with mixed precision."""
    # Low-bit quantized portion
    indices: np.ndarray          # uint8, shape (n, n_normal_channels)
    norms: np.ndarray            # float32, shape (n,)
    # Outlier channels stored at full precision
    outlier_values: np.ndarray   # float32, shape (n, n_outlier_channels)
    # Channel mapping
    outlier_mask: np.ndarray     # bool, shape (head_dim,)
    normal_mask: np.ndarray      # bool, shape (head_dim,)
    # Stats
    avg_bits: float


class MixedPrecisionQuantizer:
    """Quantizer that detects outlier channels and quantizes them at higher precision.

    Normal channels: low_bits (e.g., 3-bit TurboQuantMSE).
    Outlier channels: outlier_bits (e.g., 8-bit TurboQuantMSE) — NOT raw FP32.

    This way, 15% outliers at 8-bit + 85% normal at 3-bit = avg 3.75 bits.

    Args:
        head_dim: Total dimension per head.
        low_bits: Bit-width for normal channels (1-8).
        outlier_bits: Bit-width for outlier channels (default 8).
        outlier_threshold: Channels with RMS > threshold × median_rms are outliers.
        max_outlier_ratio: Cap on fraction of channels treated as outliers.
        seed: Random seed for TurboQuantMSE.
    """

    def __init__(self, head_dim: int, low_bits: int = 3,
                 outlier_bits: int = 8,
                 outlier_threshold: float = 3.0,
                 max_outlier_ratio: float = 0.25,
                 seed: int = 42):
        self.head_dim = head_dim
        self.low_bits = low_bits
        self.outlier_bits = outlier_bits
        self.outlier_threshold = outlier_threshold
        self.max_outlier_ratio = max_outlier_ratio
        self.seed = seed
        # Lazy init: quantizers created after we know channel splits
        self._normal_quantizer: TurboQuantMSE | None = None
        self._outlier_quantizer: TurboQuantMSE | None = None
        self._last_outlier_mask: np.ndarray | None = None

    def _detect_outliers(self, vectors: np.ndarray) -> np.ndarray:
        """Detect outlier channels based on RMS across vectors.

        Args:
            vectors: shape (n, head_dim)

        Returns:
            Boolean mask of shape (head_dim,), True = outlier channel.
        """
        # Per-channel RMS
        channel_rms = np.sqrt(np.mean(vectors ** 2, axis=0))  # (head_dim,)
        median_rms = np.median(channel_rms)

        if median_rms < 1e-8:
            return np.zeros(self.head_dim, dtype=bool)

        # Channels above threshold
        outlier_mask = channel_rms > self.outlier_threshold * median_rms

        # Cap outlier ratio
        max_outliers = int(self.head_dim * self.max_outlier_ratio)
        if outlier_mask.sum() > max_outliers:
            # Keep only top-k by RMS
            outlier_indices = np.argsort(channel_rms)[::-1][:max_outliers]
            outlier_mask = np.zeros(self.head_dim, dtype=bool)
            outlier_mask[outlier_indices] = True

        return outlier_mask

    def _ensure_quantizers(self, n_normal: int, n_outlier: int):
        """Create or recreate quantizers for normal and outlier channels."""
        if self._normal_quantizer is None or self._normal_quantizer.d != n_normal:
            if n_normal >= 3:
                self._normal_quantizer = TurboQuantMSE(n_normal, self.low_bits, self.seed)
            else:
                self._normal_quantizer = None
        if self._outlier_quantizer is None or (n_outlier >= 3 and self._outlier_quantizer.d != n_outlier):
            if n_outlier >= 3:
                self._outlier_quantizer = TurboQuantMSE(n_outlier, self.outlier_bits, self.seed + 500)
            else:
                self._outlier_quantizer = None

    def quantize(self, vectors: np.ndarray) -> QuantizedMixed:
        """Quantize with mixed precision: outliers at outlier_bits, rest at low_bits.

        Args:
            vectors: shape (n, head_dim) or (head_dim,).

        Returns:
            QuantizedMixed with separate quantized storage for outlier and normal channels.
        """
        single = vectors.ndim == 1
        if single:
            vectors = vectors[np.newaxis, :]

        vectors = vectors.astype(np.float32)

        # Detect outliers from this batch
        outlier_mask = self._detect_outliers(vectors)
        normal_mask = ~outlier_mask
        self._last_outlier_mask = outlier_mask

        n_outlier = int(outlier_mask.sum())
        n_normal = int(normal_mask.sum())
        self._ensure_quantizers(n_normal, n_outlier)

        # Quantize normal channels at low_bits
        if n_normal >= 3 and self._normal_quantizer is not None:
            normal_vectors = vectors[:, normal_mask]
            q_normal = self._normal_quantizer.quantize(normal_vectors)
            indices = q_normal.indices
            norms = q_normal.norms
        else:
            indices = np.array([], dtype=np.uint8)
            norms = np.array([], dtype=np.float32)

        # Quantize outlier channels at outlier_bits (NOT raw FP32)
        if n_outlier >= 3 and self._outlier_quantizer is not None:
            outlier_vectors = vectors[:, outlier_mask]
            q_outlier = self._outlier_quantizer.quantize(outlier_vectors)
            outlier_values = q_outlier  # Store the QuantizedMSE object
        else:
            # Too few outlier channels for TurboQuant, store raw
            outlier_values = vectors[:, outlier_mask]

        # Compute average bits
        avg_bits = (n_outlier * self.outlier_bits + n_normal * self.low_bits) / self.head_dim
        avg_bits += 32 / self.head_dim  # norm overhead

        if single:
            indices = indices[0] if indices.size > 0 else indices
            norms = norms[0] if norms.size > 0 else norms

        return QuantizedMixed(
            indices=indices,
            norms=norms,
            outlier_values=outlier_values,
            outlier_mask=outlier_mask,
            normal_mask=normal_mask,
            avg_bits=avg_bits,
        )

    def dequantize(self, q: QuantizedMixed) -> np.ndarray:
        """Reconstruct vectors from mixed precision representation."""
        from .core import QuantizedMSE

        # Determine batch size
        n_outlier = int(q.outlier_mask.sum())
        n_normal = int(q.normal_mask.sum())

        # Handle single vs batch for indices/norms
        indices = q.indices
        norms = q.norms
        single = False
        if indices.size > 0 and indices.ndim == 1 and n_normal > 1:
            # Single vector case: indices shape is (n_normal,)
            single = True
            indices = indices[np.newaxis, :]
            if np.isscalar(norms) or norms.ndim == 0:
                norms = np.array([norms], dtype=np.float32)

        # Reconstruct normal channels
        if n_normal >= 3 and indices.size > 0:
            self._ensure_quantizers(n_normal, n_outlier)
            q_mse = QuantizedMSE(indices=indices, norms=norms)
            normal_recon = self._normal_quantizer.dequantize(q_mse)
            n = normal_recon.shape[0]
        else:
            n = 1
            normal_recon = None

        result = np.zeros((n, self.head_dim), dtype=np.float32)

        if normal_recon is not None:
            result[:, q.normal_mask] = normal_recon

        # Reconstruct outlier channels
        if n_outlier >= 3 and self._outlier_quantizer is not None and hasattr(q.outlier_values, 'indices'):
            # outlier_values is a QuantizedMSE object
            outlier_recon = self._outlier_quantizer.dequantize(q.outlier_values)
            if outlier_recon.ndim == 1:
                outlier_recon = outlier_recon[np.newaxis, :]
            result[:, q.outlier_mask] = outlier_recon
        elif isinstance(q.outlier_values, np.ndarray):
            # Raw FP32 fallback (few outlier channels)
            ov = q.outlier_values
            if ov.ndim == 1:
                ov = ov[np.newaxis, :]
            result[:, q.outlier_mask] = ov

        if single:
            return result[0]
        return result

    def distortion(self, n_samples: int = 5000, seed: int = 42) -> dict:
        """Measure empirical MSE distortion with synthetic data."""
        from .utils import random_unit_vectors
        X = random_unit_vectors(n_samples, self.head_dim, seed)
        q = self.quantize(X)
        X_hat = self.dequantize(q)
        mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
        return {
            "mse": mse,
            "avg_bits": q.avg_bits,
            "n_outlier": int(q.outlier_mask.sum()),
            "n_normal": int(q.normal_mask.sum()),
        }
