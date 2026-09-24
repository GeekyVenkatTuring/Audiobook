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
