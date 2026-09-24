"""Layout analysis (RESEARCH.md §2).

Default backend: a dependency-free heuristic over PyMuPDF geometry that groups words into
line-blocks, ranks headings by relative height, and marks figure/table regions from image
and drawing rectangles. A `LayoutAnalyzer` protocol defines the pluggable interface so a
real model (DocLayout-YOLO / PP-DocLayout — see RESEARCH.md) can be dropped in without
touching downstream code.
"""

from __future__ import annotations

from statistics import median
from typing import List, Protocol

from .config import Config
from .models import Block, Page, Word, BBox


class LayoutAnalyzer(Protocol):
    """Pluggable layout interface. Implementations annotate each page with `blocks`.

    A production implementation (e.g. DocLayout-YOLO on the rendered page image) should
    fill `page.blocks` with `Block(kind=..., bbox=..., words=[...], order=...)` where kind
    is one of: text | heading | caption | figure | table | header | footer.
    """

    def analyze(self, page: Page) -> List[Block]:
        ...


def _bbox_union(boxes: List[BBox]) -> BBox:
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[2] for b in boxes)
    ys1 = max(b[3] for b in boxes)
    return (xs0, ys0, xs1, ys1)


def _overlaps(a: BBox, b: BBox, tol: float = 2.0) -> bool:
    return not (a[2] < b[0] - tol or a[0] > b[2] + tol or a[3] < b[1] - tol or a[1] > b[3] + tol)


class HeuristicLayout:
    """Fast, offline layout using only PyMuPDF word/image/drawing geometry."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze(self, page: Page) -> List[Block]:
        blocks: List[Block] = []

        # 1) Figure regions from raster images.
        for r in page.image_rects:
            blocks.append(Block(bbox=r, page=page.number, kind="figure"))

        # 2) Table candidates: large vector-drawing regions (grids of lines).
        for r in page.drawing_rects:
            w, h = r[2] - r[0], r[3] - r[1]
            if w > page.width * 0.25 and h > page.height * 0.08:
                blocks.append(Block(bbox=r, page=page.number, kind="table"))

        # 3) Group words into line-blocks (PyMuPDF's block/line numbering).
        line_groups: dict = {}
        for word in page.words:
            line_groups.setdefault((word.block, word.line), []).append(word)

        # Reference line height to distinguish headings from body text.
        line_heights = [
            (max(w.bbox[3] for w in ws) - min(w.bbox[1] for w in ws))
            for ws in line_groups.values()
            if ws
        ]
        body_h = median(line_heights) if line_heights else 0.0

        for (bno, lno), ws in sorted(line_groups.items()):
            ws_sorted = sorted(ws, key=lambda w: w.bbox[0])
            bbox = _bbox_union([w.bbox for w in ws_sorted])
            text = " ".join(w.text for w in ws_sorted)
            height = bbox[3] - bbox[1]

            kind = "text"
            # Notably taller than the median line -> heading.
            if body_h and height > body_h * 1.35:
                kind = "heading"
            # Short line starting with a caption cue -> caption.
            elif text[:7].lower().startswith(("figure", "fig.", "table")) and len(text) < 120:
                kind = "caption"
            # Text sitting inside a detected figure/table region -> absorb it.
            elif any(b.kind in ("figure", "table") and _overlaps(bbox, b.bbox) for b in blocks):
                # Keep the words with the region so captions/announcements can use them.
                for b in blocks:
                    if b.kind in ("figure", "table") and _overlaps(bbox, b.bbox):
                        b.words.extend(ws_sorted)
                        b.text = (b.text + " " + text).strip()
                        break
                continue

            blocks.append(
                Block(bbox=bbox, page=page.number, kind=kind, text=text, words=ws_sorted)
            )

        # 4) Header/footer detection by vertical position within margin bands.
        if self.config.skip_headers_footers:
            top_band = page.height * 0.06
            bot_band = page.height * 0.94
            for b in blocks:
                if b.kind == "text" and len(b.text) < 80:
                    cy = (b.bbox[1] + b.bbox[3]) / 2
                    if cy < top_band:
                        b.kind = "header"
                    elif cy > bot_band:
                        b.kind = "footer"

        blocks = order_blocks(blocks, page)
        page.blocks = blocks
        return blocks


def order_blocks(blocks: List[Block], page: Page) -> List[Block]:
    """Assign reading order.

    Simple column-aware ordering: split the page at the horizontal midpoint into left/right
    columns when content clearly occupies both, else fall back to top-to-bottom. Good enough
    for narration; a real layout model would supply logical order directly.
    """
    if not blocks:
        return blocks

    mid = page.width / 2.0
    left = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2 < mid]
    right = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2 >= mid]

    two_column = len(left) >= 3 and len(right) >= 3
    if two_column:
        ordered = sorted(left, key=lambda b: b.bbox[1]) + sorted(right, key=lambda b: b.bbox[1])
    else:
        ordered = sorted(blocks, key=lambda b: (round(b.bbox[1] / 4), b.bbox[0]))

    for i, b in enumerate(ordered):
        b.order = i
    return ordered


def get_layout_analyzer(config: Config) -> LayoutAnalyzer:
    """Factory: return the configured layout backend."""
    backend = config.layout_backend
    if backend == "heuristic":
        return HeuristicLayout(config)
    if backend == "doclayout":
        # Optional ML backend — see audiobook/backends/doclayout.py for the interface stub.
        from .backends.doclayout import DocLayoutYOLO  # noqa: WPS433 (lazy, optional dep)

        return DocLayoutYOLO(config)
    raise ValueError(f"Unknown layout_backend: {backend!r}")
