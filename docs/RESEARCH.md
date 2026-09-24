# Research notes and design decisions (2026)

This document records the options we evaluated for the three hard sub-problems
of a PDF-to-audiobook with synchronized highlighting: **layout analysis**,
**OCR for scanned PDFs**, and **TTS + word timing**. The choices baked into the
code are called out as **Decision**.

---

## 1. Layout analysis — text vs figures vs tables

The goal is to know *what* each region on a page is, so we can (a) read text in
the right order, (b) skip diagrams that have no words to read, and (c) either
skip or linearize tables.

### 1a. Born-digital PDFs — PyMuPDF native structure (our default)
PyMuPDF (`pymupdf`/`fitz`) exposes the PDF's own structure with no ML:
- `page.get_text("dict")` → **blocks → lines → spans**, each with a `bbox` and,
  for spans, font `size`/`flags`. Image blocks are marked `type == 1`. This gives
  paragraph and line boxes for free, plus font size for heading detection.
- `page.get_text("words")` → word-level boxes (useful for word-level highlight).
- `page.get_drawings()` → **vector graphics** (lines/curves/fills). Clustering
  these reveals diagrams drawn as vectors (charts, flow diagrams).
- `page.find_tables()` → a `TableFinder`; each table exposes `.bbox` and
  `.extract()` (list of row/cell strings) for linearization.
- `page.get_images()` / image blocks → raster figures.

This is fast, dependency-light, and exact for the ~majority of real-world PDFs
that are born-digital. Sources:
- PyMuPDF text-extraction appendix: https://pymupdf.readthedocs.io/en/latest/app1.html
- FAQ (blocks/words/dict modes): https://pymupdf.readthedocs.io/en/latest/faq/index.html
- Text/OCR overview: https://www.nutrient.io/blog/extract-text-from-pdf-pymupdf/

**Our classification heuristics** (see `backend/layout.py`):
- **figure** — image blocks, and clusters of vector drawings larger than ~3% of
  the page area (excludes rules/underlines and full-page background boxes).
- **table** — regions from `find_tables()`; text blocks that fall >60% inside a
  table box are absorbed into the table rather than read twice.
- **heading** — text block whose median span size ≥ 1.18× the page's body font
  size and is reasonably short.
- **caption** — short text starting with "Figure/Fig./Table/Chart/Diagram".
- **text** — everything else.
- **Reading order** — top-to-bottom, with a two-column split when a clear
  vertical gutter exists and no block spans it (heuristic; documented limit).

### 1b. Harder cases — deep-learning layout models (optional backend)
For scans, magazines, multi-column newspapers, or messy PDFs, native structure
is unreliable and a detector trained on document layout helps:
- **PP-DocLayout / PP-DocLayout-plus / PP-StructureV3 (PaddleOCR 3.0)** — current
  SOTA-ish, RT-DETR based; PP-DocLayoutV2 even predicts **reading order** with a
  pointer network. Strong on complex/multi-column/handwritten/vertical layouts.
  - https://arxiv.org/pdf/2503.17213 (PP-DocLayout)
  - https://arxiv.org/pdf/2507.05595 (PaddleOCR 3.0 tech report)
  - https://huggingface.co/PaddlePaddle/PP-DocLayout-L
- **LayoutParser** + PubLayNet/PrimaLayout-trained Detectron2 models — mature,
  easy, good general regions (text/title/list/table/figure).
- **microsoft/table-transformer** (DETR) — best-in-class **table detection +
  table-structure recognition** when tables matter a lot.

**Decision:** ship PyMuPDF native analysis as the always-on default (fast, no
weights). Deep-learning layout is documented as an optional upgrade path
(`layout_backend=ppstructure`, wired via the ML extras) rather than a hard
dependency, to keep the core install light.

### 1c. Diagram / table handling policy
- **Diagrams/figures are skipped from audio by default** — an image or vector
  chart has no narratable words, and TTS of a figure region would be noise. This
  is configurable (`read_figures`). Their **captions are read by default**
  (`read_captions`), which is usually the meaningful text about a figure.
- **Tables are skipped by default**, because naive top-to-bottom reading of a
  table is confusing. When `read_tables` is on we **linearize** row-by-row and,
  when a header row is detected, speak `header: value` pairs per cell so the
  audio is intelligible ("Row 1. Name: Alice; Score: 90.").
- To keep the listener oriented, skipped figures/tables are announced with a
  short spoken placeholder ("Figure skipped." / "Table skipped.") — toggle with
  `speak_placeholders`. The regions are still shown and highlightable in the UI.

---

## 2. OCR for scanned / image-only PDFs

**When we need it:** a page whose native text layer yields < ~20 characters is
treated as scanned/image-only and routed to OCR (per-page, so mixed documents
work). Detection = sample native extraction and check character count — the
standard, reliable approach.
- Fallback strategy references:
  https://theneuralbase.com/pdf-processing/learn/advanced/detecting-scanned-vs-digital/

**Do we need an ML OCR model to "understand layout"?** For *born-digital* PDFs,
no — PyMuPDF already provides text + geometry. OCR is needed only for scanned or
image-only pages. There, a modern model both recognizes text and (for the good
ones) preserves reading order/structure.

### Options compared (Hugging Face, 2026)
| Model | Type | Strengths | Weaknesses |
|---|---|---|---|
| **GOT-OCR2.0** (`stepfun-ai/GOT-OCR2_0`, HF-native `yonigozlan/GOT-OCR-2.0-hf`) | End-to-end OCR-2.0 VLM (~580M) | Clean reading-ordered text; formatted/structured output (math, tables); permissive license; compact; transformers-native | Returns text, not per-line boxes (page-level highlight for scans); needs torch |
| **olmOCR v2** (AllenAI) | LLM-based PDF linearizer | Excellent reading-order & layout preservation for bulk PDF→text | Heavier; geared to dataset creation; larger footprint |
| **docTR** (`mindee/doctr`) | Detection + recognition (CNN/ViT) | **Word-level boxes** (great for highlighting), fast, easy, offline | Weaker on complex layouts / handwriting than VLMs |
| **PaddleOCR / PP-StructureV3** | Detector + recognizer + structure | Strong multilingual + layout + tables; word boxes | Paddle stack heavier to install |
| **PaddleOCR-VL** (0.9B) | Compact VLM | Strong multilingual doc parsing | Newer; torch/paddle |
| **TrOCR** (`microsoft/trocr-*`) | Transformer recognizer only | Good on cropped lines/handwriting | Needs a separate detector; not full-page |
| **Tesseract** (not HF) | Classic engine | Ubiquitous, tiny, `image_to_data` gives word boxes, offline | Lower accuracy on noisy/complex scans |

Sources:
- HF open OCR models overview: https://github.com/huggingface/blog/blob/main/ocr-open-models.md
- GOT-OCR2 in transformers: https://huggingface.co/docs/transformers/main/model_doc/got_ocr2
- 2026 comparisons: https://unstract.com/blog/best-opensource-ocr-tools/ ,
  https://www.f22labs.com/blogs/ocr-models-comparison/

### Decision
- **Primary recommendation: GOT-OCR2.0** (`stepfun-ai/GOT-OCR2_0`). Best balance
  of accuracy, layout/reading-order preservation, structured output (it can emit
  formatted text incl. tables/formulas), a permissive license, a compact ~580M
  size that runs on a single modest GPU (or CPU, slowly), and a clean
  transformers-native path. It is the right default when scan quality/complexity
  matters.
- **Lightweight neural fallback: docTR** — when you need **word-level bounding
  boxes** for tighter highlighting on scanned pages, or want to avoid a VLM.
- **Always-available fallback shipped as the running default: Tesseract**
  (`pytesseract`). It requires only the `tesseract` system binary (no multi-GB
  download), and its `image_to_data` gives us line/word boxes for highlighting.
  This keeps the core install light while the HF models remain a documented,
  one-line opt-in via `requirements-ml.txt` and `ocr_provider=got_ocr|doctr`.

All OCR providers implement one interface (`backend/ocr/base.py`), so swapping is
a config change. Heavy providers are guarded so the core never needs torch.

---

## 3. Text-to-speech and word-level timing

### TTS options (local/offline first)
| Option | Quality | Offline | Notes |
|---|---|---|---|
| **Kokoro-82M** (`hexgrad/Kokoro-82M`) | High, natural | Yes | Apache-2.0, ~82M, <2GB VRAM, real-time on CPU; Python API can expose token timings |
| **Piper** | Good | Yes | Fast, small, phoneme-timed — purpose-built for read-along timing |
| **Coqui TTS** | High | Yes | Many voices; heavier |
| **pyttsx3** | Robotic | Yes | Wraps system eSpeak/NSSpeech/SAPI5; zero model download |
| **gTTS** | Good | No (network) | Google Translate voices; MP3 |

Sources:
- Kokoro word timestamps: https://ryanwelch.co.uk/blog/kokoro-word-timestamps/
- Best local TTS 2026: https://localaimaster.com/blog/best-local-tts-models
- Kokoro setup: https://localaimaster.com/blog/kokoro-tts-local-setup

### Word-level highlight timing (forced alignment)
Segment-level timing comes for free by measuring each synthesized segment's
duration. For **word-level** highlighting you either use a TTS that emits token
timings (Kokoro/Piper) or run **forced alignment** on the produced audio:
- **WhisperX** — word-level, multilingual, GPU-accelerated.
- **aeneas** — lightweight, Python, sentence/fragment-level (needs eSpeak+ffmpeg).
- **Montreal Forced Aligner (MFA)** — highest accuracy, heavier setup.
Sources: https://arxiv.org/pdf/2606.18466 (MFA / alignment state 2026),
https://arxiv.org/pdf/2509.09987 (Whisper internal aligner).

### Decision
- **Default provider: an "estimator"** that produces a valid silent WAV sized to
  each segment's estimated spoken duration (words/WPM). This is not a cop-out on
  quality — it guarantees the **entire pipeline runs in any environment** (no
  speech engine, no network, no GPU) and still yields a correct, monotonic
  timeline + playable audio container for the highlighting UI to drive.
- **Real offline speech: pyttsx3** (auto-selected when a system speech engine is
  present).
- **High quality: Kokoro-82M** as an optional extra (`tts_provider=kokoro`),
  chosen for its quality-per-byte, Apache-2.0 license, and available timing data.
- **Timing model:** per-segment start/end from measured (or estimated) durations,
  mapped to each segment's bounding boxes → line/paragraph highlighting today.
  Word-level is a documented upgrade via Kokoro token timings or WhisperX/aeneas
  forced alignment (proportional per-word split is a cheap interim option).

---

## Summary of shipped defaults
- **Layout:** PyMuPDF native (blocks/lines/words + `get_drawings` + `find_tables`).
- **Diagrams:** detected and **skipped** from audio (captions read); configurable.
- **Tables:** **skipped** by default; **linearized row-by-row** when enabled.
- **OCR:** Tesseract fallback runs out of the box; **GOT-OCR2.0** recommended and
  wired as the primary opt-in, docTR as the box-level fallback.
- **TTS:** estimator (always works) → pyttsx3 (offline) → Kokoro (quality).
