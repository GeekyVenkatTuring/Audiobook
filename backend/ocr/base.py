"""Abstract OCR provider interface.

An OCR provider takes a rendered page image (PNG bytes) and returns a list of
recognized text lines, each with a bounding box in *image pixel* coordinates
(top-left origin). The caller scales those back to PDF points.
"""
from __future__ import annotations

import abc
from typing import List, Tuple

# (text, (x0, y0, x1, y1)) with the bbox in image pixels.
OCRLine = Tuple[str, Tuple[float, float, float, float]]


class OCRProvider(abc.ABC):
    """Base class for all OCR providers."""

    name: str = "base"

    @abc.abstractmethod
    def available(self) -> bool:
        """Return True if this provider can actually run in this environment
        (dependencies importable, system binaries/weights present)."""

    @abc.abstractmethod
    def recognize(self, image_png: bytes) -> List[OCRLine]:
        """Recognize text in a PNG image, returning line-level results."""
