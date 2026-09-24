"""Pipeline orchestrator: PDF in → (audio, timing.json, page metadata) out.

Wires the stages together:

    ingest → render → layout → (ocr) → content_filter → tts → alignment → write outputs

Each stage uses the backend chosen in `Config`; defaults are fully offline. The result is
written to an output directory containing:

    audio.wav        concatenated narration
    timing.json      {audio, sample_rate, pages:[...], words:[{text,start,end,page,bbox}]}
    pages/page-N.png rendered page rasters (for the viewer / OCR)
    source.pdf       a copy of the input PDF (served to PDF.js)
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

from .alignment import get_aligner
from .config import Config
from .content_filter import build_speech_units
from .ingest import extract_document, render_pages
from .layout import get_layout_analyzer
from .models import Document, TimedWord
from .ocr import get_ocr_engine, page_needs_ocr
from .audio_utils import concat_wavs
from .tts import get_tts_engine


@dataclass
class PipelineResult:
    out_dir: str
    audio_path: str
    timing_path: str
    num_words: int
    duration_s: float


def process_pdf(
    pdf_path: str,
    out_dir: str,
    config: Optional[Config] = None,
    render: bool = True,
) -> PipelineResult:
    config = config or Config()
    os.makedirs(out_dir, exist_ok=True)
    pages_dir = os.path.join(out_dir, "pages")
    clips_dir = os.path.join(out_dir, "clips")

    # 1) Ingest words + geometry.
    document: Document = extract_document(pdf_path)

    # 2) Render page rasters (needed by the viewer, OCR, and image-based layout).
    image_paths: List[str] = []
    if render:
        image_paths = render_pages(pdf_path, pages_dir, dpi=config.render_dpi)
        for page, img in zip(document.pages, image_paths):
            page.image_path = img

    # 3) OCR any page that lacks extractable text (no-op by default).
    ocr = get_ocr_engine(config)
    for page in document.pages:
        if page_needs_ocr(page, config) and page.image_path:
            ocr_words = ocr.recognize(page, page.image_path)
            if ocr_words:
                page.words = ocr_words
                page.char_count = sum(len(w.text) for w in ocr_words)

    # 4) Layout analysis per page.
    layout = get_layout_analyzer(config)
    for page in document.pages:
        layout.analyze(page)

    # 5) Reading order + content filtering → narration units.
    units = build_speech_units(document, config)
    if not units:
        raise RuntimeError("No narratable text found in PDF (is it a scanned doc? enable OCR).")

    # 6) TTS: one clip per sentence.
    tts = get_tts_engine(config)
    clips = tts.synthesize([u.text for u in units], clips_dir)

    # 7) Concatenate audio.
    audio_path = os.path.join(out_dir, "audio.wav")
    total_duration = concat_wavs([c.wav_path for c in clips], audio_path)

    # 8) Alignment → word-level timing.
    aligner = get_aligner(config)
    timed: List[TimedWord] = aligner.align(units, clips)

    # 9) Write timing.json + metadata.
    timing = {
        "audio": "audio.wav",
        "sample_rate": config.sample_rate,
        "tts_backend": config.tts_backend,
        "alignment_backend": config.alignment_backend,
        "duration": round(total_duration, 3),
        "source_pdf": "source.pdf",
        "pages": [
            {
                "page": p.number,
                "width": p.width,
                "height": p.height,
                "image": f"pages/page-{p.number}.png" if p.image_path else None,
            }
            for p in document.pages
        ],
        "words": [w.to_dict() for w in timed],
    }
    timing_path = os.path.join(out_dir, "timing.json")
    with open(timing_path, "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=2)

    # 10) Copy the source PDF so the web viewer can render it with PDF.js.
    try:
        shutil.copyfile(pdf_path, os.path.join(out_dir, "source.pdf"))
    except OSError:
        pass

    return PipelineResult(
        out_dir=out_dir,
        audio_path=audio_path,
        timing_path=timing_path,
        num_words=len(timed),
        duration_s=total_duration,
    )
