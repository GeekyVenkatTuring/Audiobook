"""Online TTS via gTTS (Google Translate voices). Produces MP3.

Requires network access. Duration is estimated from word count (measuring MP3
duration precisely would need an extra dependency); this is adequate for
highlight timing at sentence/segment granularity.
"""
from __future__ import annotations

from .base import TTSProvider, estimate_duration


class GTTSTTSProvider(TTSProvider):
    name = "gtts"
    fmt = "mp3"

    def __init__(self, wpm: int = 165, lang: str = "en") -> None:
        self.wpm = wpm
        self.lang = lang

    def available(self) -> bool:
        try:
            import gtts  # noqa: F401

            return True
        except Exception:
            return False

    def synthesize(self, text: str, out_path: str) -> float:
        from gtts import gTTS

        gTTS(text=text or " ", lang=self.lang).save(out_path)
        # Prefer precise duration if mutagen happens to be installed.
        try:
            from mutagen.mp3 import MP3

            return float(MP3(out_path).info.length)
        except Exception:
            return estimate_duration(text, self.wpm)
