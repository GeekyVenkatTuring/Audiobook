"""Tesseract-based OCR provider (always-available lightweight fallback).

Uses pytesseract's image_to_data to get word boxes, then groups words into
lines using Tesseract's block/par/line numbering. Requires the `tesseract`
system binary; if it (or pytesseract/Pillow) is missing, `available()` returns
False and the pipeline degrades gracefully.
"""
from __future__ import annotations

import io
from collections import defaultdict
from typing import List

from .base import OCRLine, OCRProvider


class TesseractOCRProvider(OCRProvider):
    name = "tesseract"

    def __init__(self) -> None:
        self._checked = False
        self._ok = False

    def available(self) -> bool:
        if self._checked:
            return self._ok
        self._checked = True
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401

            # Probe the binary; get_tesseract_version raises if it is absent.
            import pytesseract as pt

            pt.get_tesseract_version()
            self._ok = True
        except Exception:
            self._ok = False
        return self._ok

    def recognize(self, image_png: bytes) -> List[OCRLine]:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_png))
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT
        )
        n = len(data["text"])
        # Group words by (block, par, line).
        lines = defaultdict(list)
        for i in range(n):
            word = (data["text"][i] or "").strip()
            if not word:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            lines[key].append((word, x, y, x + w, y + h))

        out: List[OCRLine] = []
        for key in sorted(lines.keys()):
            words = lines[key]
            text = " ".join(w[0] for w in words)
            x0 = min(w[1] for w in words)
            y0 = min(w[2] for w in words)
            x1 = max(w[3] for w in words)
            y1 = max(w[4] for w in words)
            out.append((text, (float(x0), float(y0), float(x1), float(y1))))
        return out
