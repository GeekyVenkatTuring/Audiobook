"""End-to-end pipeline: PDF -> layout -> segments -> audio + manifest."""
from __future__ import annotations

import json
import os
from typing import List

from .config import ProcessOptions
from .layout import Block, analyze_document
from .manifest import (
    Manifest,
    Segment,
    TYPE_FIGURE,
    TYPE_TABLE,
)
from .ocr.factory import get_ocr_provider
from .tts.factory import get_tts_provider
from .utils.audio import concat_bytes, concat_wavs

MANIFEST_VERSION = 1


def _placeholder_for(block: Block) -> str:
    if block.type == TYPE_FIGURE:
        return "Figure skipped."
    if block.type == TYPE_TABLE:
        return "Table skipped."
    return ""


def _build_segments(blocks: List[Block], opts: ProcessOptions) -> List[Segment]:
    """Turn classified blocks into ordered segments with spoken text decided."""
    segments: List[Segment] = []
    sid = 0
    for b in blocks:
        text = None
        spoken = False
        placeholder = False

        if b.read and b.text.strip():
            text = b.text.strip()
            spoken = True
        else:
            # Skipped content. Optionally announce it so the listener knows.
            ph = _placeholder_for(b)
            if ph and opts.speak_placeholders:
                text = ph
                spoken = True
                placeholder = True
            else:
                # Silent segment: kept for on-screen display, no audio.
                text = None
                spoken = False

        segments.append(
            Segment(
                id=sid,
                page=b.page,
                type=b.type,
                text=text,
                bboxes=[tuple(round(v, 2) for v in bb) for bb in (b.line_bboxes or [b.bbox])],
                spoken=spoken,
                placeholder=placeholder,
            )
        )
        sid += 1
    return segments


def process_pdf(pdf_path: str, opts: ProcessOptions, job_dir: str) -> Manifest:
    os.makedirs(job_dir, exist_ok=True)

    ocr = get_ocr_provider(opts.ocr_provider)
    pages, blocks, ocr_used, notes = analyze_document(pdf_path, opts, ocr)

    tts = get_tts_provider(opts.tts_provider, opts.wpm)
    segments = _build_segments(blocks, opts)

    # Synthesize audio per spoken segment and build the timeline.
    seg_dir = os.path.join(job_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    seg_files: List[str] = []
    cursor = 0.0
    ext = "wav" if tts.fmt == "wav" else "mp3"
    for seg in segments:
        if seg.spoken and seg.text:
            path = os.path.join(seg_dir, f"seg_{seg.id:05d}.{ext}")
            try:
                dur = tts.synthesize(seg.text, path)
            except Exception as e:  # a segment failing must not kill the job
                from .tts.base import estimate_duration
                from .utils.audio import write_silence

                path = os.path.join(seg_dir, f"seg_{seg.id:05d}.wav")
                dur = write_silence(path, estimate_duration(seg.text, opts.wpm))
                notes.append(f"TTS failed on segment {seg.id} ({e}); used silence.")
            seg.start = round(cursor, 3)
            seg.end = round(cursor + dur, 3)
            cursor = seg.end
            seg_files.append(path)
        else:
            seg.start = round(cursor, 3)
            seg.end = round(cursor, 3)

    # Concatenate segment audio into the final track.
    audio_ext = "wav"
    if seg_files and all(f.endswith(".mp3") for f in seg_files):
        audio_ext = "mp3"
    audio_file = f"audio.{audio_ext}"
    audio_path = os.path.join(job_dir, audio_file)
    if audio_ext == "wav":
        total = concat_wavs([f for f in seg_files if f.endswith(".wav")], audio_path)
    else:
        concat_bytes(seg_files, audio_path)
        total = cursor  # estimated

    manifest = Manifest(
        version=MANIFEST_VERSION,
        source_pdf=os.path.basename(pdf_path),
        options=opts.to_dict(),
        pages=pages,
        segments=segments,
        audio_file=audio_file,
        audio_format=audio_ext,
        audio_duration=round(total, 3),
        tts_provider=tts.name,
        ocr_used=ocr_used,
        notes=notes,
    )

    with open(os.path.join(job_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)

    return manifest
