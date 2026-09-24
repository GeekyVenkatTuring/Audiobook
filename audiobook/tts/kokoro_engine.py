"""Kokoro-82M TTS backend (recommended high-quality default — RESEARCH.md §4).

Kokoro (`hexgrad/Kokoro-82M`, Apache-2.0) is an 82M-parameter model that narrates faster
than real time on CPU with impressively natural prosody. This module is a thin, documented
adapter; the heavy dependency (`kokoro`) lives in requirements-ml.txt and is imported
lazily so the light core never needs it.

Enable with: pip install -r requirements-ml.txt  and  --tts kokoro
"""

from __future__ import annotations

import os
import wave
from typing import List

from ..config import Config
from .base import TTSResult


class KokoroTTS:
    def __init__(self, config: Config) -> None:
        self.config = config
        try:
            from kokoro import KPipeline  # noqa: WPS433 (optional heavy dep)
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Kokoro is not installed. `pip install -r requirements-ml.txt` "
                "(installs `kokoro`), or use a lighter tts_backend."
            ) from exc
        # 'a' = American English; choose lang code from the first letter of the voice.
        self._pipeline = KPipeline(lang_code="a")
        self._voice = None if config.voice == "default" else config.voice

    def synthesize(self, sentences: List[str], out_dir: str) -> List[TTSResult]:
        import numpy as np  # noqa: WPS433 (comes with kokoro)

        os.makedirs(out_dir, exist_ok=True)
        sr = 24000  # Kokoro's native sample rate
        results: List[TTSResult] = []
        for i, text in enumerate(sentences):
            # Kokoro yields (graphemes, phonemes, audio) chunks; concatenate them.
            chunks = [
                audio for _, _, audio in self._pipeline(text, voice=self._voice or "af_heart")
            ]
            audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            path = os.path.join(out_dir, f"clip-{i:05d}.wav")
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm)
            results.append(TTSResult(text=text, wav_path=path, duration_s=len(audio) / sr))
        return results
