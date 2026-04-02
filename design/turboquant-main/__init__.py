"""TurboQuant — Online Vector Quantization with Near-optimal Distortion Rate.

Implementation of Zandieh et al., ICLR 2026 (arXiv:2504.19874).
"""

from .core import TurboQuantMSE, TurboQuantProd, QuantizedMSE, QuantizedProd

__all__ = ["TurboQuantMSE", "TurboQuantProd", "QuantizedMSE", "QuantizedProd"]
__version__ = "0.1.0"
