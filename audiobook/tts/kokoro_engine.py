"""Kokoro-82M TTS backend (recommended high-quality default — RESEARCH.md §4).

Kokoro (`hexgrad/Kokoro-82M`, Apache-2.0) is an 82M-parameter model that narrates faster
than real time on CPU with impressively natural prosody. This module is a thin, documented
adapter; the heavy dependency (`kokoro`) lives in requirements-ml.txt and is imported
lazily so the light core never needs it.

Voices follow Kokoro's ``<lang><gender>_<name>`` convention (e.g. ``af_heart`` =
American-English female "heart"). The first letter selects the language pipeline, so we
derive ``lang_code`` from the configured voice; ``default`` maps to American English
(``af_heart``). Kokoro synthesizes at 24 kHz, which is also the pipeline's default
``sample_rate``; each sentence is written as a mono 16-bit WAV and its true duration is
reported so alignment (proportional or ctc) can place word boundaries.

Enable with: pip install -r requirements-ml.txt  and  --tts kokoro
"""

from __future__ import annotations

import os
import wave
from typing import List

from ..config import Config
from .base import TTSResult

# Kokoro's native output sample rate.
_KOKORO_SR = 24000
_DEFAULT_VOICE = "af_heart"


def _lang_code_for_voice(voice: str) -> str:
    """Kokoro derives the language pipeline from the voice's first letter.

    a=American English, b=British English, e=Spanish, f=French, h=Hindi, i=Italian,
    j=Japanese, p=Brazilian Portuguese, z=Mandarin. Fall back to American English.
    """
    first = (voice or "")[:1].lower()
    return first if first in "abefhijpz" else "a"


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

        self._voice = _DEFAULT_VOICE if config.voice == "default" else config.voice
        self._pipeline = KPipeline(lang_code=_lang_code_for_voice(self._voice))

    def synthesize(self, sentences: List[str], out_dir: str) -> List[TTSResult]:
        import numpy as np  # noqa: WPS433 (comes with kokoro)

        os.makedirs(out_dir, exist_ok=True)
        results: List[TTSResult] = []
        for i, text in enumerate(sentences):
            # Kokoro yields (graphemes, phonemes, audio) chunks; concatenate them.
            chunks = [audio for _, _, audio in self._pipeline(text, voice=self._voice)]
            if chunks:
                audio = np.concatenate([np.asarray(c, dtype="float32") for c in chunks])
            else:
                audio = np.zeros(1, dtype="float32")
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            path = os.path.join(out_dir, f"clip-{i:05d}.wav")
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_KOKORO_SR)
                wf.writeframes(pcm)
            results.append(
                TTSResult(text=text, wav_path=path, duration_s=len(audio) / _KOKORO_SR)
            )
        return results
