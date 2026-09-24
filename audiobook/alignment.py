"""Word-level timing (RESEARCH.md §5).

Produces the core output: a list of `TimedWord` mapping each spoken word to
`{start, end, page, bbox}`. The default `proportional` aligner is fully offline: it takes
each sentence's *measured* audio duration (from the TTS engine) and distributes it across
the sentence's words proportionally to their length (with a per-word floor and pause
padding at punctuation). A pluggable `Aligner` slot lets a real forced aligner
(ctc-forced-aligner / WhisperX) replace this for frame-accurate timing without changing the
output schema.
"""

from __future__ import annotations

from typing import List, Protocol

from .config import Config
from .models import SpeechUnit, TimedWord
from .tts.base import TTSResult


class Aligner(Protocol):
    """Pluggable alignment interface.

    Given the narration units and the synthesized clips (aligned 1:1, in order), return a
    flat list of TimedWord with absolute start/end times over the concatenated audio.
    """

    def align(self, units: List[SpeechUnit], clips: List[TTSResult]) -> List[TimedWord]:
        ...


def _word_weight(text: str) -> float:
    """Relative time weight for a word ~ its character length (min 1)."""
    return float(max(len(text), 1))


class ProportionalAligner:
    """Offline fallback: split each clip's duration across its words by length."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def align(self, units: List[SpeechUnit], clips: List[TTSResult]) -> List[TimedWord]:
        timed: List[TimedWord] = []
        cursor = 0.0  # absolute time over the concatenated audio

        for unit, clip in zip(units, clips):
            duration = clip.duration_s
            start_of_clip = cursor
            words = unit.words

            if not words:
                # Synthetic unit (e.g. a table announcement): still consumes audio time but
                # has no highlight box. Emit tokens spread over the clip so the UI can show
                # text scrolling if desired.
                tokens = unit.text.split()
                if tokens:
                    per = duration / len(tokens)
                    for j, tok in enumerate(tokens):
                        timed.append(TimedWord(
                            text=tok,
                            start=start_of_clip + j * per,
                            end=start_of_clip + (j + 1) * per,
                            page=unit.page,
                            bbox=None,
                        ))
                cursor += duration
                continue

            # Reserve trailing pause (if any) so the last word doesn't absorb the silence.
            pause = self.config.sentence_pause_s if unit.text.rstrip().endswith(
                (".", "!", "?", ":")) else 0.0
            speech_time = max(duration - pause, self.config.min_word_s * len(words))

            weights = [_word_weight(w.text) for w in words]
            total_w = sum(weights) or 1.0

            t = start_of_clip
            for w, weight in zip(words, weights):
                span = max(speech_time * (weight / total_w), self.config.min_word_s)
                timed.append(TimedWord(
                    text=w.text,
                    start=t,
                    end=t + span,
                    page=w.page,
                    bbox=w.bbox,
                ))
                t += span
            cursor += duration

        return timed


def get_aligner(config: Config) -> Aligner:
    """Factory: return the configured alignment backend."""
    backend = config.alignment_backend
    if backend == "proportional":
        return ProportionalAligner(config)
    if backend in ("ctc", "whisperx"):
        from .backends.forced_align import ForcedAligner  # optional heavy deps

        return ForcedAligner(config, backend)
    raise ValueError(f"Unknown alignment_backend: {backend!r}")
