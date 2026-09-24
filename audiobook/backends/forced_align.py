"""Real forced alignment (RESEARCH.md §5): ctc-forced-aligner (default) or WhisperX.

We already know the exact transcript (it is the text we sent to TTS), so a CTC forced
aligner is the right tool: it aligns known text to the synthesized audio and returns
frame-accurate word timestamps, language-agnostically, on CPU. This adapter concatenates
the per-sentence clips, runs the aligner, and maps the returned word timings back onto the
original `Word` objects (preserving each word's page + bbox), producing the same
`TimedWord` schema as the offline fallback.

Model: `MahmoudAshraf/mms-300m-1130-forced-aligner`
Install: pip install ctc-forced-aligner   (see requirements-ml.txt)
Enable:  --align ctc
"""

from __future__ import annotations

import os
from typing import List

from ..audio_utils import concat_wavs
from ..config import Config
from ..models import SpeechUnit, TimedWord
from ..tts.base import TTSResult


class ForcedAligner:
    def __init__(self, config: Config, backend: str) -> None:
        self.config = config
        self.backend = backend

    def align(self, units: List[SpeechUnit], clips: List[TTSResult]) -> List[TimedWord]:
        if self.backend == "ctc":
            return self._align_ctc(units, clips)
        return self._align_whisperx(units, clips)

    # -- ctc-forced-aligner --
    def _align_ctc(self, units, clips) -> List[TimedWord]:
        try:
            from ctc_forced_aligner import (  # noqa: WPS433
                load_alignment_model, generate_emissions, preprocess_text,
                get_alignments, get_spans, postprocess_results, load_audio,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ctc-forced-aligner not installed. `pip install ctc-forced-aligner` "
                "(see requirements-ml.txt), or use alignment_backend='proportional'."
            ) from exc

        # Flatten the original words (they carry page + bbox) in narration order.
        flat_words = [w for u in units for w in u.words]
        transcript = " ".join(w.text for w in flat_words)

        tmp_audio = os.path.join(os.path.dirname(clips[0].wav_path), "_aligned_input.wav")
        concat_wavs([c.wav_path for c in clips], tmp_audio)

        model, tokenizer = load_alignment_model()
        audio = load_audio(tmp_audio, model.dtype, model.device)
        emissions, stride = generate_emissions(model, audio)
        tokens_starred, text_starred = preprocess_text(transcript, language="eng")
        segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        word_ts = postprocess_results(text_starred, spans, stride, scores)

        timed: List[TimedWord] = []
        for w, ts in zip(flat_words, word_ts):
            timed.append(TimedWord(
                text=w.text, start=ts["start"], end=ts["end"], page=w.page, bbox=w.bbox,
            ))
        return timed

    # -- WhisperX (use only if you also need transcription of human audio) --
    def _align_whisperx(self, units, clips) -> List[TimedWord]:  # pragma: no cover
        raise NotImplementedError(
            "WhisperX path is a documented extension point; prefer 'ctc' since we already "
            "have the transcript. Implement here if aligning pre-recorded human narration."
        )
