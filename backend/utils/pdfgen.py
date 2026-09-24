"""Generate a tiny multi-element sample PDF for tests/smoke runs (no binaries
committed to the repo). Uses PyMuPDF's drawing/text APIs."""
from __future__ import annotations

import pymupdf


def make_sample_pdf(path: str) -> str:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 in points

    # Heading (large font).
    page.insert_text((72, 90), "The Little Audiobook Test", fontsize=24)

    # Body paragraph.
    body = (
        "This is a short paragraph of ordinary body text. It should be read "
        "aloud by the audiobook engine and highlighted line by line as the "
        "narration plays."
    )
    page.insert_textbox(
        pymupdf.Rect(72, 120, 523, 200), body, fontsize=12, align=0
    )

    # A vector diagram: a box with an internal shape (should be detected as a
    # figure and skipped from audio by default).
    diag = pymupdf.Rect(72, 230, 300, 360)
    page.draw_rect(diag, color=(0, 0, 0), width=1.5)
    page.draw_line(pymupdf.Point(72, 230), pymupdf.Point(300, 360), color=(0, 0, 1), width=1.2)
    page.draw_circle(pymupdf.Point(186, 295), 40, color=(1, 0, 0), width=1.2)

    # Figure caption.
    page.insert_text((72, 375), "Figure 1: A simple diagram.", fontsize=10)

    # A second paragraph.
    page.insert_textbox(
        pymupdf.Rect(72, 410, 523, 470),
        "Here is a second paragraph that continues the story after the figure.",
        fontsize=12,
    )

    # A small table drawn as grid + text (find_tables should pick it up).
    x0, y0 = 72, 500
    rows, cols = 3, 2
    cw, ch = 150, 26
    for r in range(rows + 1):
        page.draw_line(
            pymupdf.Point(x0, y0 + r * ch),
            pymupdf.Point(x0 + cols * cw, y0 + r * ch),
            width=0.8,
        )
    for c in range(cols + 1):
        page.draw_line(
            pymupdf.Point(x0 + c * cw, y0),
            pymupdf.Point(x0 + c * cw, y0 + rows * ch),
            width=0.8,
        )
    cells = [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]
    for r in range(rows):
        for c in range(cols):
            page.insert_text(
                (x0 + c * cw + 6, y0 + r * ch + 18), cells[r][c], fontsize=11
            )

    doc.save(path)
    doc.close()
    return path
