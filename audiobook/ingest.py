"""PDF ingestion + per-word bbox extraction using PyMuPDF (RESEARCH.md §1).

This is the foundation of the whole app: `page.get_text("words")` yields one tuple per
word `(x0, y0, x1, y1, word, block_no, line_no, word_no)`, giving us the bounding box that
the frontend later highlights in sync with the audio.
"""

from __future__ import annotations

import os
from typing import List

import pymupdf  # PyMuPDF; the legacy alias is `fitz`

from .models import Document, Page, Word


def _rects_to_bboxes(rects) -> List:
    return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rects]


def extract_document(pdf_path: str) -> Document:
    """Open a PDF and extract words (with bboxes), image and drawing regions per page."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    doc = pymupdf.open(pdf_path)
    pages: List[Page] = []
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            rect = page.rect

            words: List[Word] = []
            # sort=True gives natural reading order (top-to-bottom, left-to-right).
            for w in page.get_text("words", sort=True):
                x0, y0, x1, y1, text, bno, lno, wno = w
                text = text.strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        page=pno,
                        block=int(bno),
                        line=int(lno),
                        word_no=int(wno),
                    )
                )

            # Image regions: raster images placed on the page.
            image_rects = []
            for img in page.get_images(full=True):
                try:
                    for r in page.get_image_rects(img[0]):
                        image_rects.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
                except Exception:
                    continue

            # Vector drawings (lines/rects) often mean tables or diagrams.
            drawing_rects = []
            try:
                for d in page.get_drawings():
                    r = d.get("rect")
                    if r is not None:
                        drawing_rects.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
            except Exception:
                pass

            char_count = sum(len(w.text) for w in words)

            pages.append(
                Page(
                    number=pno,
                    width=float(rect.width),
                    height=float(rect.height),
                    words=words,
                    image_rects=image_rects,
                    drawing_rects=drawing_rects,
                    char_count=char_count,
                )
            )
    finally:
        doc.close()

    return Document(path=pdf_path, pages=pages)


def render_pages(pdf_path: str, out_dir: str, dpi: int = 120) -> List[str]:
    """Render each page to a PNG for the viewer. Returns the list of file paths.

    The viewer can also render the PDF directly with PDF.js; these rasters are a handy
    fallback and useful for OCR of scanned pages.
    """
    os.makedirs(out_dir, exist_ok=True)
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    paths: List[str] = []
    doc = pymupdf.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            pix = doc.load_page(pno).get_pixmap(matrix=matrix)
            path = os.path.join(out_dir, f"page-{pno}.png")
            pix.save(path)
            paths.append(path)
    finally:
        doc.close()
    return paths
