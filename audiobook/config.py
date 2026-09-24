"""Configuration for the audiobook pipeline.

A single dataclass drives behaviour. Everything the RESEARCH.md non-text policy calls
"configurable" lives here, plus the choice of pluggable backends. Defaults are chosen so
the pipeline runs offline with zero model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class Config:
    # ---- backend selection (see RESEARCH.md for the production options) ----
    # Layout backend: "heuristic" (PyMuPDF spans) | "doclayout" (DocLayout-YOLO, optional)
    layout_backend: str = "heuristic"
    # OCR backend: "none" (text-first, no OCR) | "surya" | "paddle" (optional, ML)
    ocr_backend: str = "none"
    # OCR is only attempted on pages with (almost) no extractable text.
    ocr_min_chars: int = 20
    # TTS backend: "estimated" (offline silent WAV) | "pyttsx3" (OS TTS) | "kokoro" | "piper"
    tts_backend: str = "estimated"
    # Alignment backend: "proportional" (offline) | "ctc" (ctc-forced-aligner) | "whisperx"
    alignment_backend: str = "proportional"

    # ---- non-text handling policy (RESEARCH.md §6) ----
    read_figures: bool = False          # never read figure pixels; may read their caption
    read_captions: bool = True          # read the caption near a figure/table
    table_mode: str = "announce"        # "announce" | "linearize" | "skip"
    skip_headers_footers: bool = True   # drop repeated margin text (page numbers etc.)

    # ---- speech / timing tuning ----
    words_per_minute: float = 175.0     # used by the offline "estimated" duration model
    sentence_pause_s: float = 0.35      # silence appended after sentence-ending punctuation
    min_word_s: float = 0.06            # floor so short words still get a visible highlight
    sample_rate: int = 24000            # WAV sample rate for generated audio

    # ---- rendering ----
    render_dpi: int = 120               # DPI for the page PNGs written for the viewer

    # ---- TTS voice (backend-specific; e.g. Kokoro voice id) ----
    voice: str = "default"

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)
