"""Utility functions for TurboQuant."""

import numpy as np


def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate norm and direction.

    Args:
        x: Input vector(s), shape (d,) or (n, d).

    Returns:
        (x_hat, norms) where x_hat = x / ‖x‖₂ and norms = ‖x‖₂.
    """
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            return x, np.float32(norm)
        return (x / norm).astype(np.float32), np.float32(norm)
    else:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms_flat = norms.squeeze(-1).astype(np.float32)
        safe_norms = np.where(norms < 1e-10, 1.0, norms)
        x_hat = (x / safe_norms).astype(np.float32)
        return x_hat, norms_flat


def random_unit_vectors(n: int, d: int, seed: int | None = None) -> np.ndarray:
    """Generate n random unit vectors in ℝᵈ.

    Args:
        n: Number of vectors.
        d: Dimension.
        seed: Random seed.

    Returns:
        Array of shape (n, d), each row has unit norm.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms
