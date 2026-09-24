"""Real forced alignment (RESEARCH.md §5): ctc-forced-aligner (default) or WhisperX.

We already know the exact transcript (it is the text we sent to TTS), so a CTC forced
aligner is the right tool: it aligns known text to the synthesized audio and returns
frame-accurate word timestamps, language-agnostically, on CPU. This adapter concatenates
the per-sentence clips, runs the aligner, and maps the returned word timings back onto the
original `Word` objects (preserving each word's page + bbox), producing the same
`TimedWord` schema as the offline fallback.

The transcript handed to the aligner is built from **every spoken unit** (so it matches the
audio exactly, including caption/announcement units), while each token still carries the
page + bbox of its source `Word` where one exists. Synthetic tokens (e.g. a "Table on
page N" announcement, or a caption whose text does not tokenize 1:1 with its words) keep
their timing but carry no bbox — exactly as the offline aligner treats them.

Model:  `MahmoudAshraf/mms-300m-1130-forced-aligner` (MMS-300M CTC, CPU-viable).
Install: pip install "git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"
         (see requirements-ml.txt)
Enable:  --align ctc
Env:     AUDIOBOOK_ALIGN_DEVICE (default "cpu"), AUDIOBOOK_ALIGN_MODEL (default the model
         above) let you override the runtime device and the model id without code changes.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from ..audio_utils import concat_wavs, wav_duration
from ..config import Config
from ..models import BBox, SpeechUnit, TimedWord
from ..tts.base import TTSResult

logger = logging.getLogger(__name__)

# Default aligner model (RESEARCH.md §5); overridable via AUDIOBOOK_ALIGN_MODEL.
DEFAULT_ALIGN_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"

# One token of the alignment transcript plus where (if anywhere) it appears on the page.
# (text, page, bbox) — bbox is None for synthetic tokens (announcements / caption mismatch).
_Token = Tuple[str, int, Optional[BBox]]


class ForcedAligner:
    def __init__(self, config: Config, backend: str) -> None:
        self.config = config
        self.backend = backend

    def align(self, units: List[SpeechUnit], clips: List[TTSResult]) -> List[TimedWord]:
        if self.backend == "ctc":
            return self._align_ctc(units, clips)
        return self._align_whisperx(units, clips)

    # -- token stream --------------------------------------------------------------------
    @staticmethod
    def _build_tokens(units: List[SpeechUnit]) -> List[_Token]:
        """Flatten the narration into aligner tokens that mirror the spoken audio.

        For a normal text unit ``unit.text == " ".join(w.text for w in unit.words)``, so the
        whitespace tokens line up 1:1 with the words and each carries its page + bbox. When
        that 1:1 relationship does not hold (a caption whose string differs from its words,
        or a word-less announcement), the tokens still cover the spoken words of the audio
        but carry ``None`` for the bbox.
        """
        tokens: List[_Token] = []
        for unit in units:
            words = unit.words
            text_tokens = unit.text.split()
            if words and len(text_tokens) == len(words):
                for w in words:
                    tokens.append((w.text, w.page, w.bbox))
            else:
                for tok in text_tokens:
                    tokens.append((tok, unit.page, None))
        return tokens

    # -- ctc-forced-aligner --------------------------------------------------------------
    def _align_ctc(self, units, clips) -> List[TimedWord]:
        try:
            from ctc_forced_aligner import (  # noqa: WPS433 (optional heavy dep)
                load_alignment_model,
                generate_emissions,
                preprocess_text,
                get_alignments,
                get_spans,
                postprocess_results,
                load_audio,
            )
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "ctc-forced-aligner is not installed. Install it with\n"
                '  pip install "git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"\n'
                "(see requirements-ml.txt), or use alignment_backend='proportional'."
            ) from exc

        tokens = self._build_tokens(units)
        transcript = " ".join(tok for tok, _, _ in tokens)
        if not transcript.strip():
            return []

        # Concatenate the per-sentence clips into a single file for the aligner. This is the
        # same audio the pipeline writes as audio.wav.
        tmp_audio = os.path.join(os.path.dirname(clips[0].wav_path), "_aligned_input.wav")
        concat_wavs([c.wav_path for c in clips], tmp_audio)
        audio_duration = wav_duration(tmp_audio)

        device = os.environ.get("AUDIOBOOK_ALIGN_DEVICE", "cpu")
        model_path = os.environ.get("AUDIOBOOK_ALIGN_MODEL", DEFAULT_ALIGN_MODEL)

        model, tokenizer = load_alignment_model(device, model_path=model_path)
        # load_audio resamples to the model's 16 kHz internally, regardless of clip rate.
        audio = load_audio(tmp_audio, model.dtype, model.device)
        emissions, stride = generate_emissions(model, audio)
        tokens_starred, text_starred = preprocess_text(
            transcript, romanize=True, language="eng",
        )
        segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        word_ts = postprocess_results(text_starred, spans, stride, scores)

        return self._map_timestamps(tokens, word_ts, audio_duration)

    @staticmethod
    def _map_timestamps(tokens: List[_Token], word_ts, audio_duration: float) -> List[TimedWord]:
        """Pair aligner timestamps with the token metadata, keeping timing monotonic.

        ``postprocess_results`` returns one entry per real transcript word (``<star>``
        padding already removed), so it lines up 1:1 with ``tokens`` in order. We defend
        against any count drift by pairing positionally and, if the aligner returned fewer
        entries than tokens, extending the tail with the final timestamp so every page word
        still gets a (monotonic) highlight.
        """
        if len(word_ts) != len(tokens):
            logger.warning(
                "ctc aligner returned %d timings for %d tokens; pairing positionally.",
                len(word_ts), len(tokens),
            )

        timed: List[TimedWord] = []
        prev_end = 0.0
        last_end = word_ts[-1]["end"] if word_ts else audio_duration
        for i, (tok, page, bbox) in enumerate(tokens):
            if i < len(word_ts):
                start = float(word_ts[i]["start"])
                end = float(word_ts[i]["end"])
            else:
                # Ran out of aligner timestamps: collapse remaining tokens at the tail.
                start = end = last_end
            # Enforce monotonic, sane bounds so the viewer's binary search stays valid.
            start = max(start, prev_end)
            end = max(end, start)
            end = min(end, audio_duration)
            start = min(start, audio_duration)
            timed.append(TimedWord(text=tok, start=start, end=end, page=page, bbox=bbox))
            prev_end = start
        return timed

    # -- WhisperX (use only if you also need transcription of human audio) ---------------
    def _align_whisperx(self, units, clips) -> List[TimedWord]:  # pragma: no cover
        raise NotImplementedError(
            "WhisperX path is a documented extension point; prefer 'ctc' since we already "
            "have the transcript. Implement here if aligning pre-recorded human narration."
        )
