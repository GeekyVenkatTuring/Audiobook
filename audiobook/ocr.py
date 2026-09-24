"""OCR interface for scanned / image-only pages (RESEARCH.md §3).

Default backend is a no-op: born-digital PDFs already have extractable text, so OCR is
skipped. When a page has (almost) no text and an OCR backend is configured, the pipeline
runs OCR on the rendered page image and injects the recognised words — with their
bounding boxes — into the same `Page.words` model, so highlighting works identically for
scanned pages.
"""

from __future__ import annotations

from typing import List, Protocol

from .config import Config
from .models import Page, Word


class OCREngine(Protocol):
    """Pluggable OCR interface.

    `recognize` receives the page and the path to its rendered raster (see
    `ingest.render_pages`) and returns words with bboxes in **PDF points** (the
    implementation must scale from image pixels back to points using page dimensions).
    """

    def recognize(self, page: Page, image_path: str) -> List[Word]:
        ...


class NoOpOCR:
    """Text-first default: never OCRs. Returns whatever words the page already has."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def recognize(self, page: Page, image_path: str) -> List[Word]:  # noqa: D401
        return page.words


def page_needs_ocr(page: Page, config: Config) -> bool:
    return page.char_count < config.ocr_min_chars


def get_ocr_engine(config: Config) -> OCREngine:
    """Factory: return the configured OCR backend."""
    backend = config.ocr_backend
    if backend == "none":
        return NoOpOCR(config)
    if backend in ("surya", "paddle"):
        from .backends.ocr_ml import MLDocumentOCR  # optional heavy deps

        return MLDocumentOCR(config, backend)
    raise ValueError(f"Unknown ocr_backend: {backend!r}")
