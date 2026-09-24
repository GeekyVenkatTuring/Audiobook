"""End-to-end test of the offline core pipeline (no ML, no network).

Generates a sample PDF, runs the pipeline with the default offline backends, and asserts
that audio + timing.json are produced and that timing entries carry bounding boxes.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiobook.config import Config           # noqa: E402
from audiobook.pipeline import process_pdf     # noqa: E402
from audiobook.models import SpeechUnit, Word  # noqa: E402
from audiobook.tts.base import TTSResult       # noqa: E402
from audiobook.backends.forced_align import ForcedAligner  # noqa: E402
from scripts.make_sample_pdf import make_sample  # noqa: E402


def test_offline_pipeline(tmp_path):
    pdf = tmp_path / "sample.pdf"
    make_sample(str(pdf))

    out = tmp_path / "out"
    result = process_pdf(str(pdf), str(out), config=Config(), render=True)

    # Audio + timing exist.
    assert os.path.isfile(result.audio_path)
    assert os.path.isfile(result.timing_path)
    assert result.duration_s > 0
    assert result.num_words > 0

    with open(result.timing_path, encoding="utf-8") as fh:
        timing = json.load(fh)

    assert timing["words"], "no timed words produced"
    assert timing["pages"][0]["width"] > 0

    # At least one real (non-synthetic) word carries a bbox and monotonic timing.
    boxed = [w for w in timing["words"] if w["bbox"] is not None]
    assert boxed, "no timed word has a bounding box"
    for w in boxed:
        x0, y0, x1, y1 = w["bbox"]
        assert x1 > x0 and y1 > y0
        assert w["end"] >= w["start"]

    # Timings are non-decreasing across the document.
    starts = [w["start"] for w in timing["words"]]
    assert starts == sorted(starts)


def test_ctc_aligner_token_mapping():
    """The ctc backend's transcript/token logic runs offline (no model download).

    Verifies that: real text words keep their page + bbox, a word-less announcement unit
    contributes bbox-less tokens, and the returned timings are monotonic, non-negative,
    and clamped within the audio duration — the same invariants the viewer relies on.
    """
    w1 = Word(text="Hello", bbox=(10, 20, 40, 32), page=0)
    w2 = Word(text="world.", bbox=(42, 20, 80, 32), page=0)
    units = [
        SpeechUnit(text="Hello world.", words=[w1, w2], page=0, kind="text"),
        SpeechUnit(text="Table on page 2.", words=[], page=1, kind="table"),
    ]

    aligner = ForcedAligner(Config(alignment_backend="ctc"), "ctc")

    tokens = aligner._build_tokens(units)
    assert [t[0] for t in tokens] == ["Hello", "world.", "Table", "on", "page", "2."]
    # Real words carry their bbox; the announcement tokens do not.
    assert tokens[0][2] == (10, 20, 40, 32) and tokens[1][2] == (42, 20, 80, 32)
    assert all(t[2] is None for t in tokens[2:])
    assert tokens[2][1] == 1  # announcement keeps its page

    # Fake aligner output (one entry per token) → mapping must stay monotonic + in-bounds.
    audio_duration = 3.0
    word_ts = [
        {"start": 0.0, "end": 0.5, "text": "Hello"},
        {"start": 0.5, "end": 1.0, "text": "world."},
        {"start": 1.0, "end": 1.5, "text": "Table"},
        {"start": 1.5, "end": 2.0, "text": "on"},
        {"start": 2.0, "end": 2.5, "text": "page"},
        {"start": 2.5, "end": 3.0, "text": "2."},
    ]
    timed = aligner._map_timestamps(tokens, word_ts, audio_duration)
    assert len(timed) == len(tokens)
    starts = [t.start for t in timed]
    assert starts == sorted(starts)
    for t in timed:
        assert t.end >= t.start >= 0
        assert t.end <= audio_duration + 1e-6
    assert timed[0].bbox == (10, 20, 40, 32)
    assert timed[2].bbox is None and timed[2].page == 1

    # Count drift (aligner returns fewer timings) must not drop page words or break order.
    short = aligner._map_timestamps(tokens, word_ts[:4], audio_duration)
    assert len(short) == len(tokens)
    assert [t.start for t in short] == sorted(t.start for t in short)


def test_ctc_factory_selected():
    from audiobook.alignment import get_aligner

    aligner = get_aligner(Config(alignment_backend="ctc"))
    assert isinstance(aligner, ForcedAligner)
    assert aligner.backend == "ctc"


if __name__ == "__main__":
    import tempfile

    class _P:
        def __init__(self, d):
            self._d = d

        def __truediv__(self, name):
            return os.path.join(self._d, name)

    with tempfile.TemporaryDirectory() as d:
        test_offline_pipeline(_P(d))
    print("PASS: offline pipeline test")
    test_ctc_aligner_token_mapping()
    print("PASS: ctc aligner token-mapping test")
    test_ctc_factory_selected()
    print("PASS: ctc factory test")
