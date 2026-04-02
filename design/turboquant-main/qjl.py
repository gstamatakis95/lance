"""Quantized Johnson-Lindenstrauss (QJL) — 1-bit inner product quantizer.

Q_qjl(x) = sign(S·x) where S is a random Gaussian matrix.
Provides unbiased inner product estimation with variance O(1/d).
Used as the residual corrector in TurboQuantProd.
"""

import numpy as np


def generate_projection(d: int, seed: int | None = None) -> np.ndarray:
    """Generate random Gaussian projection matrix S ∈ ℝ^(d×d).

    Args:
        d: Vector dimension.
        seed: Random seed for reproducibility.

    Returns:
        Matrix S with i.i.d. N(0,1) entries, shape (d, d), dtype float32.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, d)).astype(np.float32)


def qjl_quantize(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Quantize via random projection + sign.

    Args:
        x: Input vector(s), shape (d,) or (n, d).
        S: Projection matrix, shape (d, d).

    Returns:
        Sign array (+1/-1), dtype int8, same leading shape as x.
    """
    projected = x @ S.T  # (n, d) or (d,)
    signs = np.sign(projected).astype(np.int8)
    # Replace zeros (extremely rare) with +1
    signs[signs == 0] = 1
    return signs


def qjl_inner_product(z: np.ndarray, y: np.ndarray, S: np.ndarray,
                       residual_norm: np.ndarray) -> np.ndarray:
    """Estimate ⟨y, x_residual⟩ from QJL-quantized residual.

    ⟨y, Q⁻¹(z)⟩ = √(π/2)/d · γ · yᵀ · Sᵀ · z

    Args:
        z: QJL sign bits, shape (d,) or (n, d), dtype int8.
        y: Query vector(s), shape (d,) or (m, d).
        S: Projection matrix, shape (d, d).
        residual_norm: ‖residual‖₂, scalar or shape (n,).

    Returns:
        Estimated inner product contribution, scalar or appropriate shape.
    """
    d = S.shape[0]
    scale = np.sqrt(np.pi / 2) / d
    # y @ Sᵀ @ z → for single vectors: y @ S.T @ z
    if z.ndim == 1 and y.ndim == 1:
        return scale * residual_norm * (y @ S.T @ z.astype(np.float32))
    elif z.ndim == 2 and y.ndim == 1:
        # n quantized vectors, 1 query
        # (n, d) @ (d, d)ᵀ → not needed; compute per-vector
        Sz = z.astype(np.float32) @ S  # (n, d)
        return scale * residual_norm * (Sz @ y)  # (n,)
    else:
        # General batched case
        Sz = z.astype(np.float32) @ S  # (..., d)
        return scale * residual_norm[..., np.newaxis] * (Sz @ y.T)


def qjl_dequantize(z: np.ndarray, S: np.ndarray,
                    residual_norm: np.ndarray) -> np.ndarray:
    """Full vector reconstruction from QJL (for MSE evaluation, not typical use).

    x̃_qjl = √(π/2)/d · γ · Sᵀ · z

    Args:
        z: QJL sign bits, shape (d,) or (n, d).
        S: Projection matrix, shape (d, d).
        residual_norm: ‖residual‖₂, scalar or shape (n,).

    Returns:
        Reconstructed residual vector(s).
    """
    d = S.shape[0]
    scale = np.sqrt(np.pi / 2) / d
    z_float = z.astype(np.float32)
    if z.ndim == 1:
        return scale * residual_norm * (S.T @ z_float)
    else:
        return scale * residual_norm[:, np.newaxis] * (z_float @ S)
