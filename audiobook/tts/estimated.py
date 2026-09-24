"""Deterministic, fully-offline 'estimated-duration' TTS.

This engine does not need any model or network. It estimates how long each sentence would
take to read at `config.words_per_minute` (plus a pause after sentence-ending punctuation)
and writes a silent WAV of exactly that length. That yields real audio + a real duration
per sentence, which is everything the alignment stage needs to produce a valid timing.json
that the viewer can play against. Swap in a real engine (Kokoro/Piper) for actual voice.
"""

from __future__ import annotations

import os
import re
from typing import List

from ..audio_utils import write_silence, write_tone
from ..config import Config
from .base import TTSResult

_WORD_RE = re.compile(r"\w+")


def estimate_duration(text: str, config: Config) -> float:
    """Estimate speaking time for a sentence in seconds."""
    words = _WORD_RE.findall(text)
    n = max(len(words), 1)
    seconds = n / (config.words_per_minute / 60.0)
    if text.rstrip().endswith((".", "!", "?", ":")):
        seconds += config.sentence_pause_s
    # Floor so even a one-word sentence gets a sensible clip.
    return max(seconds, config.min_word_s * n + 0.1)


class EstimatedTTS:
    def __init__(self, config: Config) -> None:
        self.config = config
        # If AUDIOBOOK_AUDIBLE_SMOKE=1, emit a faint tone instead of silence so a human can
        # sanity-check playback timing. Off by default (pure silence = smallest files).
        self._audible = os.environ.get("AUDIOBOOK_AUDIBLE_SMOKE") == "1"

    def synthesize(self, sentences: List[str], out_dir: str) -> List[TTSResult]:
        os.makedirs(out_dir, exist_ok=True)
        results: List[TTSResult] = []
        for i, text in enumerate(sentences):
            dur = estimate_duration(text, self.config)
            path = os.path.join(out_dir, f"clip-{i:05d}.wav")
            if self._audible:
                write_tone(path, dur, self.config.sample_rate)
            else:
                write_silence(path, dur, self.config.sample_rate)
            results.append(TTSResult(text=text, wav_path=path, duration_s=dur))
        return results
