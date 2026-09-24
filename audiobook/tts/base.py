"""TTS interface shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class TTSResult:
    """One synthesized sentence."""

    text: str
    wav_path: str
    duration_s: float
    # Optional word boundaries (seconds, relative to clip start) if the engine emits them.
    # When present, alignment can skip the proportional estimate for this clip.
    word_boundaries: Optional[List[float]] = None


class TTSEngine(Protocol):
    """Pluggable TTS interface.

    Implementations synthesize each sentence to a mono 16-bit WAV at `config.sample_rate`
    and return a `TTSResult` per input string (same order).
    """

    def synthesize(self, sentences: List[str], out_dir: str) -> List[TTSResult]:
        ...
