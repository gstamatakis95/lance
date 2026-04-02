"""TurboQuant KV Cache — Drop-in layer for HuggingFace transformers Cache.

Three modes:
- "mse": Uniform TurboQuantMSE for both K and V.
- "paper": TurboQuantProd for K, TurboQuantMSE for V.
- "mixed": MixedPrecisionQuantizer for K (outlier channels at FP16), TurboQuantMSE for V.
"""

from typing import Any, Optional

import numpy as np
import torch
from transformers.cache_utils import DynamicLayer, DynamicCache

from .core import TurboQuantMSE, TurboQuantProd
from .mixed_precision import MixedPrecisionQuantizer


class TurboQuantLayer(DynamicLayer):
    """Cache layer with TurboQuant compression."""

    is_sliding = False

    def __init__(self, key_bits: int = 4, value_bits: int = 4,
                 seed: int = 42, mode: str = "mse",
                 outlier_threshold: float = 3.0,
                 max_outlier_ratio: float = 0.25):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.seed = seed
        self.mode = mode
        self.outlier_threshold = outlier_threshold
        self.max_outlier_ratio = max_outlier_ratio
        self._key_quantizer = None
        self._value_quantizer = None
        self._original_bytes = 0
        self._compressed_bits = 0
        self._avg_key_bits = float(key_bits)

    def _ensure_quantizers(self, head_dim: int):
        if self._value_quantizer is not None:
            return
        self._value_quantizer = TurboQuantMSE(head_dim, self.value_bits, self.seed)
        if self.mode == "mixed":
            self._key_quantizer = MixedPrecisionQuantizer(
                head_dim, low_bits=self.key_bits,
                outlier_threshold=self.outlier_threshold,
                max_outlier_ratio=self.max_outlier_ratio,
                seed=self.seed,
            )
        elif self.mode == "paper" and self.key_bits >= 2:
            self._key_quantizer = TurboQuantProd(head_dim, self.key_bits, self.seed)
        else:
            self._key_quantizer = TurboQuantMSE(head_dim, self.key_bits, self.seed)

    def _compress_decompress(self, states: torch.Tensor, is_key: bool) -> torch.Tensor:
        batch, heads, seq_len, head_dim = states.shape
        self._ensure_quantizers(head_dim)
        dtype = states.dtype
        bits = self.key_bits if is_key else self.value_bits

        self._original_bytes += states.numel() * states.element_size()

        flat = states.float().reshape(-1, head_dim).numpy()

        quantizer = self._key_quantizer if is_key else self._value_quantizer
        q = quantizer.quantize(flat)
        recon = quantizer.dequantize(q)

        # Track actual bits used (mixed precision has variable avg_bits)
        if is_key and self.mode == "mixed" and hasattr(q, 'avg_bits'):
            self._avg_key_bits = q.avg_bits
            self._compressed_bits += int(batch * heads * seq_len * head_dim * q.avg_bits)
        else:
            self._compressed_bits += batch * heads * seq_len * head_dim * bits

        return torch.from_numpy(recon).reshape(batch, heads, seq_len, head_dim).to(dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_lossy = self._compress_decompress(key_states, is_key=True)
        val_lossy = self._compress_decompress(value_states, is_key=False)
        return super().update(key_lossy, val_lossy, cache_kwargs)


def make_turboquant_cache(bit_width: int = 4, seed: int = 42,
                          mode: str = "mse",
                          key_bits: int | None = None,
                          value_bits: int | None = None,
                          outlier_threshold: float = 3.0,
                          max_outlier_ratio: float = 0.25) -> DynamicCache:
    """Create a DynamicCache with TurboQuant compression.

    Args:
        bit_width: Default bits per coordinate (1-8).
        seed: Random seed.
        mode: "mse" | "paper" | "mixed" (outlier-aware, recommended for large models).
        key_bits: Override bit-width for Keys.
        value_bits: Override bit-width for Values.
        outlier_threshold: For mixed mode: channels with RMS > threshold × median are outliers.
        max_outlier_ratio: For mixed mode: max fraction of channels as outliers.
    """
    kb = key_bits or bit_width
    vb = value_bits or bit_width

    class _TQLayer(TurboQuantLayer):
        def __init__(self):
            super().__init__(key_bits=kb, value_bits=vb, seed=seed,
                             mode=mode, outlier_threshold=outlier_threshold,
                             max_outlier_ratio=max_outlier_ratio)

    cache = DynamicCache.__new__(DynamicCache)
    cache.__init__()
    cache.layer_class_to_replicate = _TQLayer
    cache.layers = []
    return cache
