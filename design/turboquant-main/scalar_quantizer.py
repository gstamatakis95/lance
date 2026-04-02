"""Optimal scalar quantizer for Beta-distributed coordinates.

After random rotation, each coordinate of a unit sphere vector follows
Beta((d-1)/2, (d-1)/2) rescaled to [-1, 1]. This module computes optimal
quantization centroids via Lloyd's algorithm on the continuous distribution.
"""

import functools

import numpy as np
from scipy import integrate, special, stats


def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a single coordinate of a uniformly random unit vector in ℝᵈ.

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
    for x ∈ [-1, 1]. This is Beta((d-1)/2, (d-1)/2) rescaled from [0,1] to [-1,1].
    """
    if d <= 2:
        raise ValueError(f"d must be >= 3, got {d}")
    alpha = (d - 1) / 2
    # Use scipy's beta distribution on [0,1], then transform
    # If X ~ Beta(a, a) on [0,1], then 2X-1 ~ symmetric on [-1,1]
    # PDF on [-1,1]: f(x) = beta_pdf((x+1)/2, a, a) / 2
    return stats.beta.pdf((x + 1) / 2, alpha, alpha) / 2


def _conditional_mean(a: float, b: float, d: int) -> float:
    """E[X | a ≤ X ≤ b] under the Beta PDF on [-1, 1]."""
    num, _ = integrate.quad(lambda x: x * _beta_pdf(np.array(x), d), a, b)
    den, _ = integrate.quad(lambda x: _beta_pdf(np.array(x), d), a, b)
    if den < 1e-15:
        return (a + b) / 2
    return num / den


@functools.lru_cache(maxsize=64)
def compute_centroids(d: int, b: int, max_iter: int = 200, tol: float = 1e-10) -> tuple:
    """Compute optimal scalar quantization centroids via Lloyd's algorithm.

    Args:
        d: Vector dimension (determines the Beta distribution shape).
        b: Bit-width per coordinate (2^b centroids).
        max_iter: Maximum Lloyd iterations.
        tol: Convergence tolerance on centroid shift.

    Returns:
        (centroids, boundaries) where:
          centroids: array of 2^b optimal centroid values
          boundaries: array of 2^b + 1 decision boundaries (first=-1, last=1)
    """
    k = 2 ** b
    alpha = (d - 1) / 2

    # Initialize centroids at quantiles of the Beta distribution
    quantile_points = np.linspace(0.5 / k, 1 - 0.5 / k, k)
    # Map from Beta[0,1] quantiles to [-1,1]
    centroids = 2 * stats.beta.ppf(quantile_points, alpha, alpha) - 1

    for _ in range(max_iter):
        # Update boundaries as midpoints of adjacent centroids
        boundaries = np.empty(k + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2

        # Update centroids as conditional means
        new_centroids = np.empty(k)
        for i in range(k):
            new_centroids[i] = _conditional_mean(boundaries[i], boundaries[i + 1], d)

        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < tol:
            break

    # Recompute final boundaries
    boundaries = np.empty(k + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2

    return (
        centroids.astype(np.float32),
        boundaries.astype(np.float32),
    )


def quantize_scalar(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Quantize values to nearest centroid bin index.

    Args:
        values: Coordinate values, any shape.
        boundaries: Decision boundaries (k+1 values).

    Returns:
        Index array (uint8), same shape as values.
    """
    # searchsorted on interior boundaries gives the bin index
    indices = np.searchsorted(boundaries[1:-1], values).astype(np.uint8)
    return indices


def dequantize_scalar(indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Map bin indices back to centroid values.

    Args:
        indices: Bin index array (uint8).
        centroids: Centroid values (k values).

    Returns:
        Reconstructed values (float32), same shape as indices.
    """
    return centroids[indices]


def per_coordinate_distortion(d: int, b: int) -> float:
    """Compute expected per-coordinate MSE: E[(X - Q(X))²] under Beta PDF.

    This is 𝒞(f_X, b) from the paper. Total MSE distortion = d · 𝒞.
    For unit vectors, total distortion = d · per_coordinate_distortion.
    """
    centroids, boundaries = compute_centroids(d, b)
    k = len(centroids)
    total = 0.0
    for i in range(k):
        c = float(centroids[i])
        lo, hi = float(boundaries[i]), float(boundaries[i + 1])
        val, _ = integrate.quad(
            lambda x: (x - c) ** 2 * _beta_pdf(np.array(x), d), lo, hi
        )
        total += val
    return total
