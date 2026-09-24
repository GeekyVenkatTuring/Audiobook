# RESEARCH — PDF → Audiobook (2026)

Decision-oriented technology survey for building a "read the PDF aloud **and** highlight
each word as it is spoken (Kindle-style)" application. Every section ends with a concrete
recommendation, Hugging Face model ids where relevant, and the trade-offs that drove the
choice. The scaffold in this repo implements the **light core** of these decisions and
exposes documented, pluggable interfaces for the heavier ML pieces.

Summary of picks:

| Concern | Primary pick | Lightweight / fallback |
| --- | --- | --- |
| Word geometry (born-digital) | **PyMuPDF `get_text("words")`** | — (this is the foundation) |
| Layout analysis | **DocLayout-YOLO** (fast) or **PP-DocLayoutV3** (richer classes) | PyMuPDF block/image heuristics |
| OCR (scanned pages) | **Surya OCR** (`datalab-to/surya`) | **PaddleOCR**; TrOCR for niche/handwriting |
| TTS | **Kokoro-82M** (`hexgrad/Kokoro-82M`) | **Piper** (`rhasspy/piper-voices`) |
| Word timing | **ctc-forced-aligner** (`MahmoudAshraf/mms-300m-1130-forced-aligner`) / WhisperX | proportional char-length split (offline) |
| Non-text policy | skip figures (optionally read caption), configurable table linearization | — |

---

## 1. Born-digital text + word-level bounding boxes → **PyMuPDF (fitz)**

**Confirmed.** PyMuPDF (imported as `pymupdf`; the legacy `fitz` alias is deprecated but
still works, tested here with **v1.28.2**) is the right extractor for word geometry. It is
fast (C/MuPDF core), permissively usable (AGPL / commercial dual license), and gives us the
single most important primitive for highlighting: **per-word bounding boxes**.

- `page.get_text("words")` returns a list of tuples:
  `(x0, y0, x1, y1, "word", block_no, line_no, word_no)`.
  The first four floats are the word's rectangle in **PDF points** (origin top-left,
  y grows downward). `(x0,y0)` is the top-left corner, `(x1,y1)` the bottom-right.
  This tuple is exactly what we map each spoken word onto for the Kindle-style overlay.
- `page.get_text("dict")` / `"rawdict"` gives blocks → lines → spans with font name, size,
  flags (bold/italic) and per-span bbox — enough for a **heuristic layout pass** (headings
  via font size, body vs. caption, etc.) with zero ML.
- `page.get_text("blocks")` gives coarse text blocks and marks image blocks
  (block type `1`), useful for reading order and figure detection.
- `page.get_images()` and `page.get_drawings()` expose raster images and vector drawings
  (diagrams/tables drawn as lines) with their rectangles → the basis of "is this region
  non-text?".
- `page.rect` gives page width/height in points; `page.get_pixmap(matrix=...)` renders a
  page raster at a chosen zoom for the viewer / OCR.

**Why not alternatives:** `pdfplumber` (built on pdfminer.six) also gives word boxes but is
markedly slower and less capable at image/drawing detection; `pypdf` has no reliable
geometry. PyMuPDF wins on speed **and** on the auxiliary geometry (images/drawings) we need
for layout and non-text handling.

> Note on licensing: PyMuPDF is AGPL-3.0 (or a paid commercial license from Artifex). Fine
> for an open-source app; budget a commercial license if shipping closed-source.

Links: [get_text docs](https://pymupdf.readthedocs.io/en/latest/textpage.html) ·
[Appendix 1: text extraction details](https://pymupdf.readthedocs.io/en/latest/app1.html) ·
[repo](https://github.com/pymupdf/PyMuPDF)

---

## 2. Layout analysis (body / heading / table / figure / caption + reading order)

For born-digital PDFs, a **heuristic pass over PyMuPDF spans** already gets you far
(font-size ranking → headings; image/drawing rects → figures/tables; top-to-bottom,
left-to-right or column-aware ordering). That is the default in this scaffold. For scanned
pages, complex multi-column layouts, or real table structure you want a vision model.

Candidates evaluated (2026):

| Model | Classes | Reading order | Speed | License | Install |
| --- | --- | --- | --- | --- | --- |
| **DocLayout-YOLO** (`opendatalab/DocLayout-YOLO`, weights `juliozhao/DocLayout-YOLO-DocStructBench`) | ~10 (title, text, figure, table, caption, formula…) | no (derive yourself) | **very fast** (YOLOv10, real-time, CPU-usable) | AGPL-3.0 | easy (`doclayout-yolo` pip) |
| **PP-DocLayoutV3** (`PaddlePaddle/PP-DocLayoutV3`) | rich (title/abstract/para-title/text/table/figure/seal/formula…) | **yes** (predicts logical order, incl. skewed/curved) | fast | Apache-2.0 | via PaddleOCR/PaddleX (heavier stack) |
| **Surya layout** (`datalab-to/surya`) | strong region + **reading-order** + table structure | **yes** | medium | GPL/commercial (Datalab) | easy pip, one toolkit for layout+order+OCR |
| **Table Transformer** (`microsoft/table-transformer-structure-recognition`) | tables only (detect + structure) | n/a | medium | MIT | easy |

**Recommendation:**
- **Primary: DocLayout-YOLO** as the pluggable layout backend — best accuracy-per-millisecond,
  runs on CPU, trivial to install, covers the classes we care about (figure/table/caption
  vs. text/title). Derive reading order ourselves (column detection + top-to-bottom), which
  is cheap and good enough for narration.
- **When you need logical reading order out of the box or richer semantic classes:**
  **PP-DocLayoutV3** (Apache-2.0, and it feeds the PaddleOCR-VL parsing stack) or **Surya**
  (single toolkit that also does OCR + order, mind its GPL/commercial license).
- **For real table *structure*** (cells/rows/cols, to linearize a table into speech):
  add **Table Transformer** (MIT) on regions flagged as tables.

Trade-offs: DocLayout-YOLO is AGPL and gives boxes only (no logical order, only two "title"
granularities in some configs); PP-DocLayout needs the Paddle stack; Surya is the most
"batteries-included" but is GPL/commercial-licensed by Datalab. For a permissive, richest
one-stop pick, PP-DocLayoutV3 (Apache-2.0) is the safest.

Links: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) ·
[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) ·
[PP-DocLayout paper](https://arxiv.org/abs/2503.17213) ·
[Surya](https://github.com/datalab-to/surya) ·
[Table Transformer](https://huggingface.co/microsoft/table-transformer-structure-recognition)

---

## 3. OCR for scanned / image PDFs (Hugging Face)

Explicitly requested. The job here is **reading document *content*** (accurate text +
reading order + multilingual), not scene text. Candidates (2026):

| Model | HF id | Quality | Multilingual | Speed / size | License |
| --- | --- | --- | --- | --- | --- |
| **Surya OCR** | `datalab-to/surya` (toolkit) | high; ~83% on olmOCR-bench @ ~650M | **90+ scripts (widest)** | fast, CPU-capable | GPL / commercial |
| **PaddleOCR / PaddleOCR-VL** | `PaddlePaddle/PaddleOCR-VL` (0.9B VLM) | high, strong on structured docs | 100+ languages | fast, compact | Apache-2.0 |
| **GOT-OCR 2.0** | `stepfun-ai/GOT-OCR-2.0-hf` | **best on complex docs** (tables, formulas, mixed) | good | 580M but **wants GPU** for real-time | Apache-2.0 |
| **olmOCR v2** | `allenai/olmOCR-7B-0225` (+ newer) | excellent on messy layouts / handwriting, PDF linearization + reading order | good | **7B, GPU-heavy** | Apache-2.0 |
| **dots.ocr** | `rednote-hilab/dots.ocr` | strong, catching up on academic content | **very strong multilingual** | ~3B VLM, GPU | MIT |
| **TrOCR** | `microsoft/trocr-large-printed` / `-handwritten` | good on cropped lines | English-centric | small, needs line crops + a detector | MIT |

**Recommendation:** **Surya OCR** as the default OCR backend for reading document content.
It hits the sweet spot for this app: high accuracy, by far the **widest multilingual
coverage**, runs acceptably on CPU (GPU optional), integrates layout + reading order in the
same toolkit (so OCR output already comes ordered), and — critically for our highlighting —
returns **per-line/word bounding boxes**, so a scanned page can feed the exact same
"word + bbox" data model as a born-digital page.

- **Permissive-license alternative:** **PaddleOCR-VL** (`PaddlePaddle/PaddleOCR-VL`,
  Apache-2.0) — compact, 100+ languages, great structured-doc parsing, no GPL concerns.
- **Max accuracy on gnarly academic PDFs (GPU available):** **GOT-OCR 2.0**
  (`stepfun-ai/GOT-OCR-2.0-hf`) or **olmOCR** for handwriting / very messy scans.
- **Handwriting-only / line-level niche:** TrOCR (needs your own line detector).

Rule of thumb: traditional engines (Surya, Paddle, docTR) are lightweight and fast and great
for clean scans; VLM OCR (olmOCR, GOT, dots.ocr, Qwen2.5-VL) wins on messy layouts and
handwriting at much higher compute cost. Start with Surya/Paddle, escalate to a VLM only when
quality demands it.

Links: [Surya](https://github.com/datalab-to/surya) ·
[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) ·
[GOT-OCR2.0 (transformers)](https://huggingface.co/docs/transformers/main/model_doc/got_ocr2) ·
[olmOCR](https://huggingface.co/allenai/olmOCR-7B-0225) ·
[dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)

---

## 4. TTS — natural audiobook voice, offline-capable

| Engine | HF id | Naturalness | Footprint / speed | License | Notes |
| --- | --- | --- | --- | --- | --- |
| **Kokoro-82M** | `hexgrad/Kokoro-82M` | **high for its size**, natural prosody/pacing | 82M, **faster than real-time on CPU** | **Apache-2.0** | 54 voices, 8 languages; built for narration |
| **Piper** | `rhasspy/piper-voices` | decent, a bit robotic | tiny, ~0.03 RTF, ~40 ms first audio | MIT | fastest, smallest; great fallback |
| **XTTS-v2** | `coqui/XTTS-v2` | gold-standard cloning, very expressive | large, GPU-preferred | **CPML (non-commercial only)** | Coqui shut down; no commercial license |

**Recommendation:**
- **Default: Kokoro-82M** (`hexgrad/Kokoro-82M`). Best quality-for-size, Apache-2.0 (safe to
  ship), runs faster-than-real-time on CPU, purpose-built for narration → ideal audiobook
  default.
- **Lightweight fallback: Piper** (`rhasspy/piper-voices`, MIT). Ultra-fast and tiny for
  low-resource machines or quick previews; more robotic but perfectly intelligible.
- **Avoid for production: XTTS-v2** — superb voice cloning but **CPML non-commercial** and
  now unmaintained (Coqui defunct). Fine for personal/experimental cloning only.

This scaffold ships two runnable engines that need **no model download**: a deterministic
`estimated` engine (silent WAV sized to the estimated speaking duration — for offline
pipeline/timing verification) and an optional `pyttsx3` engine (OS TTS). Kokoro/Piper are
documented drop-in backends behind the same interface.

Links: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) ·
[Piper](https://github.com/rhasspy/piper) ·
[XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

---

## 5. Word-level highlight synchronization

Two ways to get per-word timestamps:

1. **TTS that emits word boundaries.** Some engines expose word-boundary callbacks/events
   (e.g. SAPI/pyttsx3 `onWord`, Azure/edge-tts `WordBoundary`). When available this is the
   cheapest, most accurate route — no second model. Kokoro/Piper don't emit reliable
   word events, so for them use forced alignment.
2. **Forced alignment** of the synthesized (or original) audio against the known text:

| Tool | How | Accuracy | Cost |
| --- | --- | --- | --- |
| **ctc-forced-aligner** (`MahmoudAshraf/mms-300m-1130-forced-aligner`) | CTC + Viterbi, **language-agnostic** (MMS-300M), aligns known text to audio | high, purpose-built for alignment | 300M, CPU-ok |
| **WhisperX** | Whisper transcribe → discard timings → wav2vec2 phoneme forced alignment + VAD | high, but re-transcribes (we already know the text) | Whisper + aligner, heavier |
| **NeMo Forced Aligner (NFA)** | CTC ASR + Viterbi, token/word/segment timestamps | high | NeMo stack |
| **aeneas** | DTW of TTS'd text vs. audio | decent on audiobook audio, not word-first | light, old |

**Recommendation:** Use **forced alignment with `ctc-forced-aligner`
(`MahmoudAshraf/mms-300m-1130-forced-aligner`)** as the real backend: we already have the
exact transcript (it's the text we sent to TTS), so a CTC forced aligner is faster and more
accurate than WhisperX's transcribe-then-align. It's language-agnostic and CPU-viable. Use
**WhisperX** only if you also need transcription (e.g. aligning pre-recorded human audio).
Caveat from the literature: even strong aligners place only ~2/3 of word boundaries within
20 ms, and a "word boundary" isn't a crisp event — good enough for a smooth reading
highlight.

**Offline fallback (implemented as the default here):** distribute each **sentence's measured
audio duration across its words proportionally to word length** (character or syllable count,
with a small floor per word and pause padding at punctuation). This needs no model, is
deterministic, and produces a perfectly usable first-pass highlight. When a real TTS reports
its clip duration per sentence, this fallback is quite accurate; swap in `ctc-forced-aligner`
later for frame-accurate timing without changing the data model
(`{text, start, end, page, bbox}` per word).

Links: [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) ·
[WhisperX](https://github.com/m-bain/whisperX) ·
[NeMo NFA](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/api.html) ·
[aeneas](https://github.com/readbeyond/aeneas)

---

## 6. Non-text handling policy (figures / diagrams / tables)

Reading raw figure/table pixels aloud is noise. Policy (all **configurable**, defaults shown):

**Figures / diagrams / images** — *skip in audio by default.*
- Detect via layout class `figure`/`image` (DocLayout-YOLO / PP-DocLayout) or PyMuPDF
  `get_images()` / image blocks / `get_drawings()`.
- Optionally read the **caption** (nearest text block classified `caption`, or the text block
  immediately below/above the figure rect). Config: `figures.read_caption = true|false`.
- Never OCR text *inside* a diagram by default (it's usually labels, not prose).

**Tables** — three configurable modes:
- `announce` (default): say "Table on page N" (optionally the table caption) and skip the
  cells — keeps narration flowing.
- `linearize`: read **row by row, left to right** ("Row 1: <c1>, <c2>, …"), using table
  structure from Table Transformer / PP-Structure when available, else the raw text order.
- `skip`: ignore entirely (no announcement).
- Config: `tables.mode = announce|linearize|skip`.

**Headers / footers / page numbers** — skip by default (detect via repeated position across
pages or margin bands). **Hyphenation** at line breaks — join `wor-\nd` → `word` before TTS
while keeping both sub-boxes for highlighting.

Rationale: an audiobook should read the *narrative* smoothly; visual-only content is either
skipped or summarized by its caption, and tabular data is opt-in because linearized tables can
be long and tedious. Everything is a config flag so power users can turn it up.

---

### How this maps to the scaffold

- **Implemented (light core):** PyMuPDF word-bbox ingestion (§1), heuristic layout (§2),
  content filtering + non-text policy (§6), an offline `estimated` TTS + optional OS TTS (§4),
  and the proportional-duration alignment fallback (§5) — the pipeline runs end-to-end with
  **no ML downloads**.
- **Pluggable, documented interfaces (heavy ML, optional):** `LayoutAnalyzer` →
  DocLayout-YOLO/PP-DocLayout, `OCREngine` → Surya/PaddleOCR, `TTSEngine` → Kokoro/Piper,
  `Aligner` → ctc-forced-aligner/WhisperX. Enable via `requirements-ml.txt` and config; the
  word→`{start,end,page,bbox}` data model stays identical.
