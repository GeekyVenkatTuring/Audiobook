"""Abstract TTS provider interface.

A provider synthesizes a text string to an audio file and reports the resulting
duration in seconds. The pipeline calls it once per readable segment, then
concatenates the segment files and builds the timeline from the durations.
"""
from __future__ import annotations

import abc


def estimate_duration(text: str, wpm: int = 165) -> float:
    """Rough speech duration from word count, plus a little pause padding."""
    words = max(1, len((text or "").split()))
    base = (words / max(60, wpm)) * 60.0
    return base + 0.35  # small inter-segment pause


class TTSProvider(abc.ABC):
    name: str = "base"
    fmt: str = "wav"  # 'wav' or 'mp3'

    @abc.abstractmethod
    def available(self) -> bool:
        """Whether this provider can run in the current environment."""

    @abc.abstractmethod
    def synthesize(self, text: str, out_path: str) -> float:
        """Write audio for `text` to `out_path`; return duration in seconds."""
