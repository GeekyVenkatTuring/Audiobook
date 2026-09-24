"""Configuration and processing options."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Root of the repo (two levels up from this file: backend/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where processed jobs (uploaded PDF, audio, manifest) are stored.
DATA_DIR = Path(os.environ.get("AUDIOBOOK_DATA_DIR", REPO_ROOT / "data" / "jobs"))

# Directory of the static frontend served by FastAPI.
FRONTEND_DIR = REPO_ROOT / "frontend"


@dataclass
class ProcessOptions:
    """User-tunable options controlling what gets read and how.

    Attributes:
        read_figures:   Read figure/diagram regions? Default False (images have
                        no words to read). Their captions can still be read.
        read_tables:    Read tables? If True, tables are linearized row-by-row.
                        If False they are skipped (optionally with a spoken
                        placeholder).
        read_captions:  Read captions attached to figures/tables. Default True.
        speak_placeholders: Speak a short placeholder ("Figure skipped",
                        "Table skipped") where content is skipped, so the
                        listener knows something was there. Default True.
        ocr_provider:   'auto' | 'tesseract' | 'got_ocr' | 'doctr' | 'none'.
        tts_provider:   'auto' | 'estimator' | 'pyttsx3' | 'gtts' | 'kokoro'.
        wpm:            Words-per-minute assumption for the estimator TTS and
                        for proportional word-timing splits.
        ocr_min_chars:  A page with fewer than this many extractable characters
                        is treated as scanned and routed through OCR.
    """

    read_figures: bool = False
    read_tables: bool = False
    read_captions: bool = True
    speak_placeholders: bool = True
    ocr_provider: str = "auto"
    tts_provider: str = "auto"
    wpm: int = 165
    ocr_min_chars: int = 20

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_form(cls, form: dict) -> "ProcessOptions":
        """Build options from raw string form values (all optional)."""

        def as_bool(v, default):
            if v is None:
                return default
            return str(v).strip().lower() in {"1", "true", "yes", "on"}

        def as_int(v, default):
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        d = cls()
        return cls(
            read_figures=as_bool(form.get("read_figures"), d.read_figures),
            read_tables=as_bool(form.get("read_tables"), d.read_tables),
            read_captions=as_bool(form.get("read_captions"), d.read_captions),
            speak_placeholders=as_bool(
                form.get("speak_placeholders"), d.speak_placeholders
            ),
            ocr_provider=(form.get("ocr_provider") or d.ocr_provider),
            tts_provider=(form.get("tts_provider") or d.tts_provider),
            wpm=as_int(form.get("wpm"), d.wpm),
            ocr_min_chars=as_int(form.get("ocr_min_chars"), d.ocr_min_chars),
        )
