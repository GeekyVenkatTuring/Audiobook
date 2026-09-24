"""Pipeline tests. Run with `pytest -q` (or `python -m backend.smoke`).

These exercise the full born-digital path with the always-available default
providers (estimator TTS, no OCR needed) so they run anywhere.
"""
import os
import tempfile

import pytest

from backend.config import ProcessOptions
from backend.layout import linearize_table
from backend.pipeline import process_pdf
from backend.utils.audio import wav_duration
from backend.utils.pdfgen import make_sample_pdf


@pytest.fixture()
def sample_pdf():
    d = tempfile.mkdtemp(prefix="ab_test_")
    p = os.path.join(d, "sample.pdf")
    make_sample_pdf(p)
    return d, p


def test_defaults_skip_figures_and_tables(sample_pdf):
    workdir, pdf = sample_pdf
    m = process_pdf(pdf, ProcessOptions(), os.path.join(workdir, "j1")).to_dict()

    assert len(m["pages"]) == 1
    types = {s["type"] for s in m["segments"]}
    assert "heading" in types
    assert "text" in types
    assert "figure" in types
    assert "table" in types

    # By default figures/tables are not read as content (they may be announced).
    fig = next(s for s in m["segments"] if s["type"] == "figure")
    assert fig["placeholder"] is True  # "Figure skipped."
    tbl = next(s for s in m["segments"] if s["type"] == "table")
    assert tbl["placeholder"] is True


def test_timeline_is_monotonic_and_audio_exists(sample_pdf):
    workdir, pdf = sample_pdf
    job = os.path.join(workdir, "j2")
    m = process_pdf(pdf, ProcessOptions(), job).to_dict()

    last = 0.0
    for s in m["segments"]:
        assert s["end"] >= s["start"] >= last - 1e-6
        last = s["end"]

    audio = os.path.join(job, m["audio_file"])
    assert os.path.exists(audio)
    assert wav_duration(audio) > 0
    # concatenated duration should match the manifest within a frame or two
    assert abs(wav_duration(audio) - m["audio_duration"]) < 0.1


def test_read_tables_linearizes(sample_pdf):
    workdir, pdf = sample_pdf
    m = process_pdf(
        pdf, ProcessOptions(read_tables=True), os.path.join(workdir, "j3")
    ).to_dict()
    tbl = next(s for s in m["segments"] if s["type"] == "table")
    assert tbl["spoken"] is True
    assert "Alice" in (tbl["text"] or "")
    assert "Bob" in (tbl["text"] or "")


def test_bboxes_present_for_display(sample_pdf):
    workdir, pdf = sample_pdf
    m = process_pdf(pdf, ProcessOptions(), os.path.join(workdir, "j4")).to_dict()
    for s in m["segments"]:
        assert isinstance(s["bboxes"], list) and len(s["bboxes"]) >= 1
        for bb in s["bboxes"]:
            assert len(bb) == 4


def test_linearize_table_helper():
    rows = [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]
    out = linearize_table(rows)
    assert "Name: Alice" in out
    assert "Score: 90" in out
