"""High-quality local TTS via Kokoro-82M (optional, requirements-ml.txt).

Kokoro (hexgrad/Kokoro-82M, Apache-2.0) is a compact, natural-sounding neural
TTS that runs offline on CPU. It writes 24kHz mono audio. We resample to the
project's 16-bit PCM WAV format via `soundfile`/`numpy`.
"""
from __future__ import annotations

from ..utils.audio import wav_duration
from .base import TTSProvider


class KokoroTTSProvider(TTSProvider):
    name = "kokoro"
    fmt = "wav"

    def __init__(self, wpm: int = 165, voice: str = "af_heart") -> None:
        self.wpm = wpm
        self.voice = voice
        self._pipeline = None

    def available(self) -> bool:
        try:
            import kokoro  # noqa: F401
            import soundfile  # noqa: F401
            import numpy  # noqa: F401

            return True
        except Exception:
            return False

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code="a")  # American English

    def synthesize(self, text: str, out_path: str) -> float:
        import numpy as np
        import soundfile as sf

        self._load()
        chunks = []
        for _, _, audio in self._pipeline(text, voice=self.voice):
            chunks.append(audio)
        if chunks:
            wav = np.concatenate(chunks)
        else:
            wav = np.zeros(1, dtype="float32")
        # Kokoro outputs 24kHz float; write as 16-bit PCM WAV.
        sf.write(out_path, wav, 24000, subtype="PCM_16")
        return wav_duration(out_path)
