"""Offline TTS via pyttsx3 (uses system eSpeak / NSSpeech / SAPI5).

Produces real speech with no network. Requires a system speech engine; if none
is present, `available()` returns False and the factory falls back to the
estimator provider.
"""
from __future__ import annotations

import os

from ..utils.audio import wav_duration
from .base import TTSProvider, estimate_duration


class Pyttsx3TTSProvider(TTSProvider):
    name = "pyttsx3"
    fmt = "wav"

    def __init__(self, wpm: int = 165) -> None:
        self.wpm = wpm
        self._checked = False
        self._ok = False

    def available(self) -> bool:
        if self._checked:
            return self._ok
        self._checked = True
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.stop()
            self._ok = True
        except Exception:
            self._ok = False
        return self._ok

    def synthesize(self, text: str, out_path: str) -> float:
        import pyttsx3

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", int(self.wpm))
        except Exception:
            pass
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        engine.stop()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 44:
            try:
                return wav_duration(out_path)
            except Exception:
                pass
        # Engine failed to write a usable file; fall back to a silent estimate.
        from ..utils.audio import write_silence

        return write_silence(out_path, estimate_duration(text, self.wpm))
