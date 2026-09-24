"""Shared data model passed between pipeline stages.

Coordinates are always in **PDF points** (origin top-left, y grows downward), matching
PyMuPDF. The frontend scales these boxes by the PDF.js viewport to draw the highlight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class Word:
    """A single word with its geometry on the page."""

    text: str
    bbox: BBox
    page: int
    block: int = 0
    line: int = 0
    word_no: int = 0


@dataclass
class Block:
    """A layout region (paragraph, heading, figure, table, caption ...)."""

    bbox: BBox
    page: int
    kind: str = "text"          # text | heading | caption | figure | table | header | footer
    text: str = ""
    words: List[Word] = field(default_factory=list)
    order: int = 0              # reading order within the document


@dataclass
class Page:
    number: int
    width: float                # points
    height: float               # points
    words: List[Word] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    image_rects: List[BBox] = field(default_factory=list)
    drawing_rects: List[BBox] = field(default_factory=list)
    char_count: int = 0
    image_path: Optional[str] = None   # rendered PNG, filled at render time


@dataclass
class Document:
    path: str
    pages: List[Page] = field(default_factory=list)


@dataclass
class SpeechUnit:
    """A chunk of text to be spoken (usually a sentence), with its source words.

    Words carry the geometry we need to highlight; some units (e.g. a "Table on page 3"
    announcement) may have no backing words and therefore no highlight boxes.
    """

    text: str
    words: List[Word] = field(default_factory=list)
    page: int = 0
    kind: str = "text"


@dataclass
class TimedWord:
    """The final output atom: a spoken word with when/where it appears."""

    text: str
    start: float               # seconds
    end: float                 # seconds
    page: int
    bbox: Optional[BBox]       # None for synthetic words (e.g. announcements)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "page": self.page,
            "bbox": [round(c, 2) for c in self.bbox] if self.bbox else None,
        }
