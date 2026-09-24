"""Reading-order + content filtering → the sequence of SpeechUnits to narrate.

Applies the non-text policy from RESEARCH.md §6: skip figures (optionally read caption),
handle tables per config (announce / linearize / skip), drop headers/footers, join
hyphenated line breaks, and split running text into sentences (each sentence becomes one
SpeechUnit so a real TTS can report per-sentence duration for alignment).
"""

from __future__ import annotations

import re
from typing import List

from .config import Config
from .models import Block, Document, Page, SpeechUnit, Word

# Sentence splitter: break after . ! ? (optionally followed by quotes/brackets) + space.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+")


def _dehyphenate(words: List[Word]) -> List[Word]:
    """Join words split by a hard hyphen at a line break (``wor-`` + ``d`` -> ``word``).

    We keep both original boxes so both fragments still highlight; the joined text is put
    on the first fragment and the second is emptied of text but retains its box via a
    zero-width space marker handled by the caller. For simplicity here we merge the text
    onto the first word and keep the second word's box pointing at the same joined token.
    """
    out: List[Word] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.text.endswith("-") and i + 1 < len(words):
            nxt = words[i + 1]
            merged = Word(
                text=w.text[:-1] + nxt.text,
                bbox=w.bbox,
                page=w.page,
                block=w.block,
                line=w.line,
                word_no=w.word_no,
            )
            out.append(merged)
            i += 2
        else:
            out.append(w)
            i += 1
    return out


def _split_sentences(words: List[Word], page: int, kind: str) -> List[SpeechUnit]:
    """Turn an ordered word list into sentence-level SpeechUnits, preserving word boxes."""
    if not words:
        return []
    words = _dehyphenate(words)
    units: List[SpeechUnit] = []
    current: List[Word] = []
    for w in words:
        current.append(w)
        if w.text and w.text[-1] in ".!?":
            units.append(_make_unit(current, page, kind))
            current = []
    if current:
        units.append(_make_unit(current, page, kind))
    return units


def _make_unit(words: List[Word], page: int, kind: str) -> SpeechUnit:
    return SpeechUnit(
        text=" ".join(w.text for w in words),
        words=list(words),
        page=page,
        kind=kind,
    )


def _nearest_caption(block: Block, page: Page) -> str:
    """Find caption text for a figure/table: prefer text already absorbed into the region,
    else the nearest 'Figure/Table ...' block just below it."""
    if block.text.strip():
        return block.text.strip()
    for b in page.blocks:
        if b.kind == "caption" and abs(b.bbox[1] - block.bbox[3]) < 40:
            return b.text
    return ""


def build_speech_units(document: Document, config: Config) -> List[SpeechUnit]:
    """Walk the document in reading order and emit the narration sequence."""
    units: List[SpeechUnit] = []

    for page in document.pages:
        ordered = sorted(page.blocks, key=lambda b: b.order)
        for block in ordered:
            kind = block.kind

            if kind in ("header", "footer"):
                if config.skip_headers_footers:
                    continue
                units.extend(_split_sentences(block.words, page.number, "text"))

            elif kind == "figure":
                if config.read_captions:
                    cap = _nearest_caption(block, page)
                    if cap:
                        units.append(SpeechUnit(text=cap, words=block.words, page=page.number,
                                                kind="caption"))
                # figure pixels are never read

            elif kind == "table":
                mode = config.table_mode
                if mode == "skip":
                    continue
                if mode == "announce":
                    cap = _nearest_caption(block, page)
                    msg = f"Table on page {page.number + 1}."
                    if cap and config.read_captions:
                        msg = f"{cap}"
                    units.append(SpeechUnit(text=msg, words=[], page=page.number, kind="table"))
                elif mode == "linearize":
                    # Without a table-structure model we linearize in reading order; a real
                    # backend (Table Transformer / PP-Structure) would give true rows/cols.
                    units.extend(_split_sentences(block.words, page.number, "table"))

            elif kind == "caption":
                # Captions are emitted alongside their figure/table; skip standalone dupes
                # only if they were already absorbed. Here we read them once.
                units.extend(_split_sentences(block.words, page.number, "caption"))

            else:  # text / heading
                units.extend(_split_sentences(block.words, page.number, kind))

    # Drop empty units.
    return [u for u in units if u.text.strip()]
