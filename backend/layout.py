"""Layout analysis for born-digital and scanned PDFs using PyMuPDF (fitz).

Responsibilities:
  * Extract text as blocks/lines WITH bounding boxes for born-digital PDFs.
  * Detect images (get_text image blocks) and vector graphics (get_drawings)
    as figure/diagram regions.
  * Detect tables via PyMuPDF's find_tables() and linearize them on demand.
  * Classify each block: text | heading | figure | table | caption.
  * Decide a reading order and which blocks to READ vs SKIP.
  * Route scanned / image-only pages through an OCR provider.

Coordinates in the returned blocks are PDF points with a top-left origin
(matching pdf.js viewport space), so the frontend can scale them directly.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pymupdf  # PyMuPDF (>=1.24 exposes the `pymupdf` name; `fitz` is legacy)

from .config import ProcessOptions
from .manifest import (
    BBox,
    PageInfo,
    TYPE_CAPTION,
    TYPE_FIGURE,
    TYPE_HEADING,
    TYPE_TABLE,
    TYPE_TEXT,
)
from .ocr.base import OCRProvider

_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?|table|tbl\.?|chart|diagram)\b", re.I)


@dataclass
class Block:
    """A classified region of a page, in PDF-point (top-left origin) coords."""

    page: int
    type: str
    bbox: BBox
    text: str = ""
    # Line-level rects for finer highlighting (falls back to [bbox]).
    line_bboxes: List[BBox] = field(default_factory=list)
    read: bool = True  # whether this block should be spoken
    table_rows: Optional[List[List[str]]] = None  # linearized table cells


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _contained_ratio(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    ia = _area(inner)
    if ia <= 0:
        return 0.0
    return _intersection(inner, outer) / ia


def _union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _expand(b: BBox, m: float) -> BBox:
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def _overlaps(a: BBox, b: BBox) -> bool:
    return _intersection(a, b) > 0


# --------------------------------------------------------------------------- #
# Vector-drawing clustering (diagram detection)
# --------------------------------------------------------------------------- #
def _cluster_drawings(rects: List[BBox], gap: float = 6.0) -> List[BBox]:
    """Union-find cluster of drawing rects that touch/nearly touch."""
    clusters: List[BBox] = []
    for r in rects:
        er = _expand(r, gap)
        merged = None
        for i, c in enumerate(clusters):
            if _overlaps(er, c):
                merged = i
                break
        if merged is None:
            clusters.append(r)
        else:
            clusters[merged] = _union(clusters[merged], r)
    # A second pass to merge clusters that became adjacent after growth.
    changed = True
    while changed:
        changed = False
        out: List[BBox] = []
        for c in clusters:
            hit = None
            for i, o in enumerate(out):
                if _overlaps(_expand(c, gap), o):
                    hit = i
                    break
            if hit is None:
                out.append(c)
            else:
                out[hit] = _union(out[hit], c)
                changed = True
        clusters = out
    return clusters


# --------------------------------------------------------------------------- #
# Table linearization
# --------------------------------------------------------------------------- #
def linearize_table(rows: List[List[str]]) -> str:
    """Turn table cells into a readable, row-by-row spoken string."""
    if not rows:
        return ""
    header = [c.strip() for c in rows[0]] if rows else []
    lines: List[str] = []
    has_header = header and any(header)
    body = rows[1:] if has_header else rows
    for r_i, row in enumerate(body, start=1):
        cells = [(c or "").strip() for c in row]
        if not any(cells):
            continue
        if has_header and len(header) == len(cells):
            pairs = [
                f"{h}: {v}" for h, v in zip(header, cells) if h or v
            ]
            lines.append(f"Row {r_i}. " + "; ".join(pairs) + ".")
        else:
            lines.append(f"Row {r_i}. " + "; ".join(c for c in cells if c) + ".")
    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Reading order
# --------------------------------------------------------------------------- #
def _reading_order(blocks: List[Block], page_width: float) -> List[Block]:
    """Order blocks top-to-bottom, splitting into two columns when a clear
    vertical gutter exists near the page middle. Heuristic but robust."""
    if not blocks:
        return blocks
    mid = page_width / 2.0
    left = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2.0 < mid]
    right = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2.0 >= mid]
    # Only treat as two-column if both sides are populated and neither side
    # has blocks that span across the gutter (wide blocks -> single column).
    spans_gutter = any(b.bbox[0] < mid < b.bbox[2] and
                       (b.bbox[2] - b.bbox[0]) > page_width * 0.55
                       for b in blocks)
    if left and right and not spans_gutter and len(left) >= 2 and len(right) >= 2:
        left.sort(key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
        right.sort(key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
        return left + right
    return sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))


# --------------------------------------------------------------------------- #
# Born-digital page analysis
# --------------------------------------------------------------------------- #
def _analyze_digital_page(page, page_index: int, opts: ProcessOptions) -> List[Block]:
    blocks: List[Block] = []

    # 1) Tables first, so we can suppress text blocks that live inside them.
    table_bboxes: List[BBox] = []
    try:
        tf = page.find_tables()
        for t in tf.tables:
            tb = tuple(float(v) for v in t.bbox)  # type: ignore
            table_bboxes.append(tb)
            rows = None
            try:
                rows = t.extract()
            except Exception:
                rows = None
            read_it = opts.read_tables
            text = linearize_table(rows) if (rows and read_it) else ""
            if read_it and not text:
                read_it = False
            blocks.append(
                Block(
                    page=page_index,
                    type=TYPE_TABLE,
                    bbox=tb,
                    text=text,
                    line_bboxes=[tb],
                    read=read_it,
                    table_rows=rows,
                )
            )
    except Exception:
        pass

    # 2) Text + image blocks from the dict extraction.
    data = page.get_text("dict")
    span_sizes: List[float] = []
    text_blocks_raw = []
    for b in data.get("blocks", []):
        bbox = tuple(float(v) for v in b["bbox"])
        if b.get("type") == 1:  # image block -> figure
            blocks.append(
                Block(
                    page=page_index,
                    type=TYPE_FIGURE,
                    bbox=bbox,
                    text="",
                    line_bboxes=[bbox],
                    read=False,
                )
            )
            continue
        # text block
        lines = b.get("lines", [])
        line_bboxes: List[BBox] = []
        parts: List[str] = []
        for ln in lines:
            lb = tuple(float(v) for v in ln["bbox"])
            line_bboxes.append(lb)
            spans = ln.get("spans", [])
            for sp in spans:
                span_sizes.append(float(sp.get("size", 0)))
            parts.append("".join(sp.get("text", "") for sp in spans))
        text = " ".join(p.strip() for p in parts if p.strip()).strip()
        if not text:
            continue
        text_blocks_raw.append((bbox, text, line_bboxes))

    body_size = statistics.median(span_sizes) if span_sizes else 10.0

    # 3) Classify text blocks; drop those swallowed by a table.
    for bbox, text, line_bboxes in text_blocks_raw:
        if any(_contained_ratio(bbox, tb) > 0.6 for tb in table_bboxes):
            continue  # part of a table, already represented

        btype = TYPE_TEXT
        # Heading: notably larger text and short-ish, single/few lines.
        avg_len = len(text)
        if span_sizes:
            # recompute this block's dominant size
            pass
        # crude: if the whole block text is short and larger than body
        block_sizes = [
            float(sp.get("size", 0))
            for ln in _lines_for_bbox(data, bbox)
            for sp in ln.get("spans", [])
        ]
        blk_size = statistics.median(block_sizes) if block_sizes else body_size
        if blk_size >= body_size * 1.18 and avg_len <= 120:
            btype = TYPE_HEADING
        if _CAPTION_RE.match(text) and avg_len <= 220:
            btype = TYPE_CAPTION

        read_it = True
        if btype == TYPE_CAPTION and not opts.read_captions:
            read_it = False
        blocks.append(
            Block(
                page=page_index,
                type=btype,
                bbox=bbox,
                text=text,
                line_bboxes=line_bboxes or [bbox],
                read=read_it,
            )
        )

    # 4) Vector-graphics diagrams via drawings clustering.
    try:
        draw_rects = [tuple(float(v) for v in d["rect"]) for d in page.get_drawings()]
    except Exception:
        draw_rects = []
    if draw_rects:
        page_area = _area((0, 0, page.rect.width, page.rect.height))
        for cl in _cluster_drawings(draw_rects):
            if _area(cl) < page_area * 0.03:
                continue  # too small: likely rules/underlines, not a diagram
            # skip if it mostly coincides with a detected table
            if any(_contained_ratio(cl, tb) > 0.5 for tb in table_bboxes):
                continue
            # skip if it's essentially the full page (background box)
            if _area(cl) > page_area * 0.92:
                continue
            blocks.append(
                Block(
                    page=page_index,
                    type=TYPE_FIGURE,
                    bbox=cl,
                    text="",
                    line_bboxes=[cl],
                    read=False,
                )
            )

    return _apply_read_rules(blocks, opts)


def _lines_for_bbox(data: dict, bbox: BBox):
    for b in data.get("blocks", []):
        if b.get("type") == 1:
            continue
        if tuple(float(v) for v in b["bbox"]) == bbox:
            return b.get("lines", [])
    return []


def _apply_read_rules(blocks: List[Block], opts: ProcessOptions) -> List[Block]:
    """Apply figure/table read toggles uniformly."""
    for b in blocks:
        if b.type == TYPE_FIGURE:
            b.read = opts.read_figures
        elif b.type == TYPE_TABLE:
            # read flag already reflects whether we have linearized text
            b.read = b.read and opts.read_tables
        elif b.type == TYPE_CAPTION:
            b.read = opts.read_captions
    return blocks


# --------------------------------------------------------------------------- #
# Scanned page analysis (OCR)
# --------------------------------------------------------------------------- #
def _analyze_scanned_page(
    page, page_index: int, opts: ProcessOptions, ocr: Optional[OCRProvider]
) -> Tuple[List[Block], bool, Optional[str]]:
    """Returns (blocks, ocr_used, note)."""
    if ocr is None or not ocr.available():
        note = (
            f"Page {page_index + 1} appears scanned/image-only but no OCR "
            f"provider is available; it was skipped."
        )
        placeholder = Block(
            page=page_index,
            type=TYPE_TEXT,
            bbox=(0, 0, page.rect.width, page.rect.height),
            text=f"Page {page_index + 1} could not be read: no OCR available.",
            line_bboxes=[(0, 0, page.rect.width, page.rect.height)],
            read=opts.speak_placeholders,
        )
        return [placeholder], False, note

    zoom = 2.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    lines = ocr.recognize(pix.tobytes("png"))  # list[(text, (x0,y0,x1,y1) px)]
    blocks: List[Block] = []
    for text, bbox_px in lines:
        text = (text or "").strip()
        if not text:
            continue
        bbox = tuple(v / zoom for v in bbox_px)
        blocks.append(
            Block(
                page=page_index,
                type=TYPE_TEXT,
                bbox=bbox,
                text=text,
                line_bboxes=[bbox],
                read=True,
            )
        )
    return blocks, True, None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def analyze_document(
    pdf_path: str, opts: ProcessOptions, ocr: Optional[OCRProvider]
) -> Tuple[List[PageInfo], List[Block], bool, List[str]]:
    """Analyze a PDF into ordered pages + classified, read-ordered blocks.

    Returns (pages, ordered_blocks, ocr_used, notes).
    """
    doc = pymupdf.open(pdf_path)
    pages: List[PageInfo] = []
    ordered_blocks: List[Block] = []
    ocr_used = False
    notes: List[str] = []

    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            w, h = float(page.rect.width), float(page.rect.height)
            raw_text = page.get_text("text") or ""
            scanned = len(raw_text.strip()) < opts.ocr_min_chars

            if scanned:
                blocks, used, note = _analyze_scanned_page(page, i, opts, ocr)
                ocr_used = ocr_used or used
                if note:
                    notes.append(note)
                pages.append(PageInfo(index=i, width=w, height=h, ocr=used))
            else:
                blocks = _analyze_digital_page(page, i, opts)
                pages.append(PageInfo(index=i, width=w, height=h, ocr=False))

            ordered_blocks.extend(_reading_order(blocks, w))
    finally:
        doc.close()

    return pages, ordered_blocks, ocr_used, notes
