"""Select an OCR provider from an options string."""
from __future__ import annotations

from typing import Optional

from .base import OCRProvider
from .tesseract_provider import TesseractOCRProvider


def get_ocr_provider(name: str) -> Optional[OCRProvider]:
    """Return an OCR provider, or None if OCR is disabled/unavailable.

    'auto'      -> Tesseract if available, else None (pipeline skips scans).
    'tesseract' -> Tesseract (fallback provider).
    'got_ocr'   -> GOT-OCR2.0 (requires requirements-ml.txt).
    'doctr'     -> docTR (requires requirements-ml.txt).
    'none'      -> None (never OCR).
    """
    name = (name or "auto").lower()
    if name == "none":
        return None

    if name in ("got_ocr", "got", "gotocr"):
        from .hf_provider import GOTOCRProvider

        p = GOTOCRProvider()
        return p if p.available() else _fallback()

    if name == "doctr":
        from .hf_provider import DocTROCRProvider

        p = DocTROCRProvider()
        return p if p.available() else _fallback()

    if name == "tesseract":
        return TesseractOCRProvider()

    # auto
    return _fallback()


def _fallback() -> Optional[OCRProvider]:
    t = TesseractOCRProvider()
    return t if t.available() else None
