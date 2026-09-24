"""Estimator TTS provider - the always-available, zero-dependency default.

It does not synthesize speech. Instead it produces a valid silent WAV sized to
the estimated spoken duration of each segment. This guarantees the full
pipeline (layout -> segments -> timeline -> manifest -> playable audio file)
runs in ANY environment, even with no speech engine, network, or GPU.

Swap in `pyttsx3`, `gtts`, or `kokoro` for real speech (see factory + README).
"""
from __future__ import annotations

from ..utils.audio import write_silence
from .base import TTSProvider, estimate_duration


class EstimatorTTSProvider(TTSProvider):
    name = "estimator"
    fmt = "wav"

    def __init__(self, wpm: int = 165) -> None:
        self.wpm = wpm

    def available(self) -> bool:
        return True

    def synthesize(self, text: str, out_path: str) -> float:
        dur = estimate_duration(text, self.wpm)
        return write_silence(out_path, dur)
