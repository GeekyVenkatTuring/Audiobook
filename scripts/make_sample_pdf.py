"""Generate a small, born-digital sample PDF for testing the pipeline.

The PDF has a heading, two body paragraphs, a figure caption, and a simple drawn box (to
exercise figure/table heuristics) — enough to verify word bboxes, filtering, and timing.

Usage: python scripts/make_sample_pdf.py sample.pdf
"""

from __future__ import annotations

import sys

import pymupdf


def make_sample(path: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # US Letter

    page.insert_text((72, 90), "The Tortoise and the Hare", fontsize=22, fontname="helv")

    body1 = (
        "A hare was making fun of the tortoise one day for being so slow. "
        "Do you ever get anywhere, he asked with a mocking laugh. "
        "Yes, replied the tortoise, and I get there sooner than you think."
    )
    body2 = (
        "The hare ran almost out of sight at once, but soon stopped and lay down to nap. "
        "The tortoise plodded on and plodded on. When the hare awoke, the tortoise had won."
    )
    page.insert_textbox(pymupdf.Rect(72, 130, 540, 240), body1, fontsize=12, fontname="helv")
    page.insert_textbox(pymupdf.Rect(72, 250, 540, 340), body2, fontsize=12, fontname="helv")

    # A drawn rectangle to simulate a figure/diagram region.
    page.draw_rect(pymupdf.Rect(72, 360, 300, 480), color=(0, 0, 0), width=1)
    page.insert_text((72, 500), "Figure 1. A race between two unlikely rivals.",
                     fontsize=10, fontname="helv")

    doc.save(path)
    doc.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    make_sample(out)
