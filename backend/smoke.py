"""End-to-end smoke test: generate a tiny PDF and run the full pipeline with
the default lightweight providers. Prints a summary and exits non-zero on
failure.

    python -m backend.smoke
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

from .config import ProcessOptions
from .pipeline import process_pdf
from .utils.pdfgen import make_sample_pdf


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="audiobook_smoke_")
    pdf_path = os.path.join(workdir, "sample.pdf")
    make_sample_pdf(pdf_path)
    print(f"[smoke] generated sample PDF: {pdf_path}")

    opts = ProcessOptions()  # defaults: skip figures/tables, estimator/tesseract auto
    manifest = process_pdf(pdf_path, opts, os.path.join(workdir, "job"))

    d = manifest.to_dict()
    audio_path = os.path.join(workdir, "job", d["audio_file"])

    print(f"[smoke] tts provider     : {d['tts_provider']}")
    print(f"[smoke] pages            : {len(d['pages'])}")
    print(f"[smoke] segments         : {len(d['segments'])}")
    spoken = [s for s in d["segments"] if s["spoken"]]
    print(f"[smoke] spoken segments  : {len(spoken)}")
    print(f"[smoke] audio duration   : {d['audio_duration']}s")
    print(f"[smoke] audio file       : {audio_path} "
          f"({os.path.getsize(audio_path)} bytes)")
    print(f"[smoke] ocr used         : {d['ocr_used']}")
    if d["notes"]:
        print(f"[smoke] notes            : {d['notes']}")

    # Show a couple of segments for eyeballing.
    for s in d["segments"][:6]:
        t = (s["text"] or "")[:48].replace("\n", " ")
        print(f"    #{s['id']:>2} p{s['page']} {s['type']:<8} "
              f"[{s['start']:>6.2f}-{s['end']:>6.2f}] {t!r}")

    # Assertions.
    ok = True
    if len(d["pages"]) < 1:
        print("[smoke] FAIL: no pages"); ok = False
    if len(spoken) < 1:
        print("[smoke] FAIL: no spoken segments"); ok = False
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) <= 44:
        print("[smoke] FAIL: audio file missing/empty"); ok = False
    types = {s["type"] for s in d["segments"]}
    if "heading" not in types:
        print("[smoke] WARN: heading not detected")
    if "figure" not in types:
        print("[smoke] WARN: figure not detected")
    if "table" not in types:
        print("[smoke] WARN: table not detected")
    # timeline must be monotonic
    last = 0.0
    for s in d["segments"]:
        if s["start"] < last - 1e-6:
            print("[smoke] FAIL: non-monotonic timeline"); ok = False; break
        last = s["end"]

    print("[smoke] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
