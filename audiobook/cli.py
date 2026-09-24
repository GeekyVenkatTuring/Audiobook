"""Command-line entrypoint: turn a PDF into audio + timing.json.

Examples:
    python -m audiobook.cli sample.pdf -o out/
    python -m audiobook.cli book.pdf -o out/ --tts kokoro --align ctc --table-mode linearize
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .pipeline import process_pdf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audiobook",
        description="PDF → narrated audiobook with word-level highlight timing.",
    )
    p.add_argument("pdf", help="path to the input PDF")
    p.add_argument("-o", "--out", default="output", help="output directory (default: output)")

    p.add_argument("--layout", default="heuristic", choices=["heuristic", "doclayout"],
                   help="layout backend (default: heuristic, offline)")
    p.add_argument("--ocr", default="none", choices=["none", "surya", "paddle"],
                   help="OCR backend for scanned pages (default: none)")
    p.add_argument("--tts", default="estimated",
                   choices=["estimated", "pyttsx3", "kokoro", "piper"],
                   help="TTS backend (default: estimated, offline silent audio)")
    p.add_argument("--align", default="proportional",
                   choices=["proportional", "ctc", "whisperx"],
                   help="alignment backend (default: proportional, offline)")
    p.add_argument("--voice", default="default", help="voice id / model path for the TTS backend")

    p.add_argument("--table-mode", default="announce", choices=["announce", "linearize", "skip"],
                   help="how to handle tables (RESEARCH.md §6)")
    p.add_argument("--read-figures", action="store_true",
                   help="read text found inside figures (off by default)")
    p.add_argument("--no-captions", action="store_true", help="do not read figure/table captions")
    p.add_argument("--wpm", type=float, default=175.0,
                   help="words per minute for the offline estimated-duration model")
    p.add_argument("--no-render", action="store_true",
                   help="skip rendering page PNGs (faster; viewer still works via PDF.js)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(
        layout_backend=args.layout,
        ocr_backend=args.ocr,
        tts_backend=args.tts,
        alignment_backend=args.align,
        voice=args.voice,
        table_mode=args.table_mode,
        read_figures=args.read_figures,
        read_captions=not args.no_captions,
        words_per_minute=args.wpm,
    )

    result = process_pdf(args.pdf, args.out, config=config, render=not args.no_render)

    print(f"OK  wrote {result.num_words} timed words over {result.duration_s:.1f}s")
    print(f"    audio : {result.audio_path}")
    print(f"    timing: {result.timing_path}")
    print(f"    open the web viewer with:  python -m server.app --out {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
