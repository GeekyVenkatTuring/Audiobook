"""OS text-to-speech via pyttsx3 (optional, offline).

pyttsx3 drives the platform speech engine (SAPI5 on Windows, NSSpeechSynthesizer on macOS,
espeak/espeak-ng on Linux). It needs a system speech backend installed (e.g.
`apt-get install espeak-ng`). This gives a *real* audible smoke test without downloading
any ML model. If the backend or library is missing, construction raises a clear error and
the pipeline can fall back to the 'estimated' engine.
"""

from __future__ import annotations

import os
from typing import List

from ..audio_utils import wav_duration
from ..config import Config
from .base import TTSResult


class Pyttsx3TTS:
    def __init__(self, config: Config) -> None:
        self.config = config
        try:
            import pyttsx3  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "pyttsx3 is not installed. `pip install pyttsx3` and install a system "
                "speech engine (e.g. espeak-ng), or use tts_backend='estimated'."
            ) from exc
        self._pyttsx3 = pyttsx3

    def synthesize(self, sentences: List[str], out_dir: str) -> List[TTSResult]:
        os.makedirs(out_dir, exist_ok=True)
        engine = self._pyttsx3.init()
        results: List[TTSResult] = []
        for i, text in enumerate(sentences):
            path = os.path.join(out_dir, f"clip-{i:05d}.wav")
            engine.save_to_file(text, path)
            engine.runAndWait()
            if not os.path.isfile(path):  # pragma: no cover - backend dependent
                raise RuntimeError(
                    "pyttsx3 did not produce audio; a system speech backend "
                    "(e.g. espeak-ng) is likely missing."
                )
            results.append(TTSResult(text=text, wav_path=path, duration_s=wav_duration(path)))
        return results
