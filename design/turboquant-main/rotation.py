"""Random rotation matrix generation and application.

A random orthogonal rotation transforms unit sphere vectors so that each
coordinate follows a known Beta distribution, enabling precomputed optimal
scalar quantization (data-oblivious).
"""

import numpy as np


def generate_rotation(d: int, seed: int | None = None) -> np.ndarray:
    """Generate a random orthogonal rotation matrix via QR decomposition.

    Args:
        d: Vector dimension.
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix Π ∈ ℝ^(d×d), dtype float32.
    """
    rng = np.random.default_rng(seed)
    # Random Gaussian matrix
    G = rng.standard_normal((d, d)).astype(np.float32)
    # QR decomposition
    Q, R = np.linalg.qr(G)
    # Ensure proper rotation (det = +1) by fixing sign ambiguity
    # Standard trick: multiply columns of Q by sign of diagonal of R
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return Q.astype(np.float32)


def rotate(x: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply rotation: y = x @ Πᵀ (supports batched input).

    Args:
        x: Input vector(s), shape (d,) or (n, d).
        rotation: Orthogonal matrix Π, shape (d, d).

    Returns:
        Rotated vector(s), same shape as x.
    """
    return x @ rotation.T


def inverse_rotate(y: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply inverse rotation: x̃ = y @ Π.

    For orthogonal Π, Π⁻¹ = Πᵀ, so inverse_rotate(y) = y @ Π.
    """
    return y @ rotation
