"""Piper TTS backend (lightweight fallback — RESEARCH.md §4).

Piper (`rhasspy/piper-voices`, MIT) is a tiny, extremely fast neural TTS. This adapter
shells out to the `piper` binary (or the `piper-tts` Python package) which must be
installed along with a downloaded voice `.onnx` model. Kept optional and lazy.

Enable with: pip install piper-tts  (and download a voice), then  --tts piper --voice <model.onnx>
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List

from ..audio_utils import wav_duration
from ..config import Config
from .base import TTSResult


class PiperTTS:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._bin = shutil.which("piper")
        if not self._bin:
            raise RuntimeError(
                "`piper` binary not found. Install piper-tts and a voice model, "
                "or use a different tts_backend."
            )
        # config.voice must point at a Piper voice .onnx file for this backend.
        self._model = None if config.voice == "default" else config.voice
        if not self._model or not os.path.isfile(self._model):
            raise RuntimeError(
                "Piper requires --voice pointing to a downloaded voice .onnx model."
            )

    def synthesize(self, sentences: List[str], out_dir: str) -> List[TTSResult]:
        os.makedirs(out_dir, exist_ok=True)
        results: List[TTSResult] = []
        for i, text in enumerate(sentences):
            path = os.path.join(out_dir, f"clip-{i:05d}.wav")
            subprocess.run(
                [self._bin, "--model", self._model, "--output_file", path],
                input=text.encode("utf-8"),
                check=True,
            )
            results.append(TTSResult(text=text, wav_path=path, duration_s=wav_duration(path)))
        return results
