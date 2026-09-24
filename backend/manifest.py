"""Data models for the audiobook manifest.

The manifest is the single artifact the frontend consumes. It describes each
page's geometry, the ordered readable/skipped segments with their bounding
boxes and per-segment audio timing, plus metadata about the generated audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# A bounding box in PDF point coordinates, top-left origin: (x0, y0, x1, y1).
BBox = Tuple[float, float, float, float]

# Block/segment classification labels.
TYPE_TEXT = "text"
TYPE_HEADING = "heading"
TYPE_FIGURE = "figure"
TYPE_TABLE = "table"
TYPE_CAPTION = "caption"


@dataclass
class PageInfo:
    index: int
    width: float
    height: float
    # True when the page had no usable text layer and was OCR'd.
    ocr: bool = False


@dataclass
class Segment:
    """One readable (or intentionally skipped) unit of the document."""

    id: int
    page: int
    type: str
    # The text that is actually spoken. None/empty for silently-skipped figures.
    text: Optional[str]
    # One or more rects to highlight while this segment plays (line-level for
    # text, region-level for figures/tables).
    bboxes: List[BBox] = field(default_factory=list)
    # Audio timeline, seconds from the start of the concatenated audio.
    start: float = 0.0
    end: float = 0.0
    # Whether this segment contributes audio. Skipped-and-silent = False.
    spoken: bool = True
    # Set when text is a generated placeholder (e.g. "Table skipped").
    placeholder: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Manifest:
    version: int
    source_pdf: str
    options: dict
    pages: List[PageInfo] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    audio_file: str = "audio.wav"
    audio_format: str = "wav"
    audio_duration: float = 0.0
    tts_provider: str = ""
    ocr_used: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source_pdf": self.source_pdf,
            "options": self.options,
            "pages": [asdict(p) for p in self.pages],
            "segments": [asdict(s) for s in self.segments],
            "audio_file": self.audio_file,
            "audio_format": self.audio_format,
            "audio_duration": self.audio_duration,
            "tts_provider": self.tts_provider,
            "ocr_used": self.ocr_used,
            "notes": self.notes,
        }
