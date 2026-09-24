# PDF → Audiobook

Turn a PDF into a narrated audiobook that **highlights each word as it is spoken**,
Kindle-style. Give it a PDF; it extracts the text with per-word bounding boxes, decides
what to read (skipping figures, handling tables per policy), synthesizes speech, computes
word-level timing, and serves a web viewer that renders the PDF and moves a highlight box
over the current word in sync with the audio.

The **core pipeline runs fully offline** on born-digital PDFs with no ML downloads. The
heavier pieces (advanced layout, OCR for scans, premium neural TTS, real forced alignment)
are **optional, pluggable backends** with documented interfaces — see
[`RESEARCH.md`](RESEARCH.md) for the 2026 technology survey and the recommended model for
each slot.

---

## Architecture

```
                ┌────────────┐   words + bboxes (PDF points, top-left origin)
   PDF ───────► │  ingest    │  PyMuPDF: page.get_text("words") + images/drawings
                └─────┬──────┘
                      ▼
                ┌────────────┐   heuristic (default) │ DocLayout-YOLO (optional)
                │  layout    │  classify blocks: text/heading/figure/table/caption + order
                └─────┬──────┘
                      ▼
                ┌────────────┐   text-first (default) │ Surya / PaddleOCR (optional)
                │   OCR      │  only for pages with no extractable text
                └─────┬──────┘
                      ▼
                ┌────────────┐   skip figures (read caption?), tables: announce|linearize|skip
                │content_flt │  → ordered list of sentence "SpeechUnits" (words keep bboxes)
                └─────┬──────┘
                      ▼
                ┌────────────┐   estimated silent WAV (default) │ pyttsx3 │ Kokoro │ Piper
                │    TTS      │  one clip per sentence, with measured duration
                └─────┬──────┘
                      ▼
                ┌────────────┐   proportional split (default) │ ctc-forced-aligner (optional)
                │ alignment  │  → TimedWord {text, start, end, page, bbox}
                └─────┬──────┘
                      ▼
        audio.wav  +  timing.json  +  pages/*.png  +  source.pdf
                      ▼
                ┌────────────┐
                │ web viewer │  PDF.js renders page; overlay box = bbox × renderScale
                └────────────┘
```

Every arrow is a small module in `audiobook/`; every "(optional)" backend lives in
`audiobook/backends/` and is imported lazily so the core never needs its dependencies.

### Why the highlight math is simple
PyMuPDF word boxes use a **top-left origin with y growing downward — the same as an HTML
canvas**. A PDF.js canvas rendered at scale `S` (pixels per PDF point) therefore maps a word
box directly: `pixel = point × S`, with no y-flip. The viewer looks up the active word by
`audio.currentTime` (binary search over the sorted `words` array) and positions one overlay
`<div>`.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) make a tiny sample PDF
python scripts/make_sample_pdf.py sample.pdf

# 2) build the audiobook (fully offline, no downloads)
python -m audiobook.cli sample.pdf -o output

# 3) view it: render + play + word highlighting
python -m server.app --out output
#   → open http://127.0.0.1:8000/?src=/out
```

Or run the server and **upload** a PDF from the browser:

```bash
uvicorn server.app:app --reload      # open http://127.0.0.1:8000/ and click "Open PDF"
```

### Outputs (`output/`)
| File | What |
| --- | --- |
| `audio.wav` | concatenated narration |
| `timing.json` | `{audio, sample_rate, duration, pages:[{page,width,height,image}], words:[{text,start,end,page,bbox}]}` |
| `pages/page-N.png` | rendered page rasters (for the viewer / OCR) |
| `source.pdf` | copy of the input, served to PDF.js |

---

## CLI

```
python -m audiobook.cli PDF -o OUT [options]

  --layout {heuristic,doclayout}     layout backend (default heuristic)
  --ocr    {none,surya,paddle}       OCR for scanned pages (default none)
  --tts    {estimated,pyttsx3,kokoro,piper}   speech backend (default estimated)
  --align  {proportional,ctc,whisperx}        word-timing backend (default proportional)
  --voice  VOICE                     voice id / model path for the TTS backend
  --table-mode {announce,linearize,skip}      table policy (default announce)
  --read-figures                     read text inside figures (off by default)
  --no-captions                      don't read figure/table captions
  --wpm FLOAT                        speaking rate for the offline duration model (175)
  --no-render                        skip page PNGs (viewer still works via PDF.js)
```

### Two offline TTS modes (no downloads)
- **`estimated`** (default): writes a **silent** WAV sized to the estimated speaking
  duration of each sentence. Deterministic; lets you generate and verify the full pipeline
  and `timing.json` with zero dependencies. Set `AUDIOBOOK_AUDIBLE_SMOKE=1` to emit a faint
  tone instead of silence for an audible playback/timing sanity check.
- **`pyttsx3`**: real OS voice (needs a system speech engine, e.g. `apt-get install espeak-ng`).

---

## Enabling the heavy ML backends

Install the extras you want (each pulls in torch / model weights):

```bash
pip install -r requirements-ml.txt   # then uncomment the specific lines you need
```

| Slot | Enable | Recommended model (HF id) |
| --- | --- | --- |
| **Premium TTS** | `--tts kokoro` | `hexgrad/Kokoro-82M` (Apache-2.0); fallback `--tts piper` (`rhasspy/piper-voices`) |
| **Advanced layout** | `--layout doclayout` | `juliozhao/DocLayout-YOLO-DocStructBench` |
| **OCR (scanned PDFs)** | `--ocr surya` (or `paddle`) | Surya (`datalab-to/surya`) / `PaddlePaddle/PaddleOCR-VL` |
| **Real forced alignment** | `--align ctc` | `MahmoudAshraf/mms-300m-1130-forced-aligner` |

Each backend implements the same small interface (`LayoutAnalyzer`, `OCREngine`,
`TTSEngine`, `Aligner`) and emits the identical `{text,start,end,page,bbox}` word schema, so
nothing downstream changes. Rationale, trade-offs and links are in
[`RESEARCH.md`](RESEARCH.md).

Example with the full ML stack:
```bash
python -m audiobook.cli book.pdf -o out \
    --layout doclayout --ocr surya --tts kokoro --align ctc --table-mode linearize
```

---

## Non-text policy (configurable)

Defaults follow [`RESEARCH.md` §6](RESEARCH.md):
- **Figures / diagrams**: never read the pixels; optionally read the nearest caption
  (`--no-captions` to disable, `--read-figures` to also read text inside figures).
- **Tables**: `--table-mode announce` (say "Table on page N", default) / `linearize`
  (read row-by-row) / `skip`.
- **Headers / footers / page numbers**: dropped by default.
- **Hyphenated line breaks** are re-joined before speaking.

---

## Project layout

```
audiobook/                package: the pipeline
  config.py               all tunables + backend selection
  models.py               Word / Block / Page / SpeechUnit / TimedWord
  ingest.py               PyMuPDF word-bbox extraction + page rendering
  layout.py               heuristic layout + LayoutAnalyzer interface
  content_filter.py       reading order + non-text policy → SpeechUnits
  ocr.py                  OCREngine interface + text-first no-op default
  tts/                    TTSEngine interface + estimated/pyttsx3/kokoro/piper
  alignment.py            Aligner interface + proportional offline default
  pipeline.py             orchestrator: PDF → audio + timing.json + metadata
  cli.py                  command-line entrypoint
  backends/               optional heavy ML adapters (lazy imports)
server/
  app.py                  FastAPI: viewer + upload/process + asset serving
  static/                 index.html, viewer.js (PDF.js), style.css
scripts/make_sample_pdf.py
tests/test_pipeline.py    offline end-to-end test
requirements.txt          light core
requirements-ml.txt       optional heavy backends
RESEARCH.md               2026 technology survey + recommendations
```

## Testing

```bash
pip install pytest && pytest -q
# or, dependency-free:
python tests/test_pipeline.py
```
The test builds a sample PDF, runs the offline pipeline, and asserts that `audio.wav` +
`timing.json` are produced and that timed words carry bounding boxes with monotonic timing.

---

## Limitations & roadmap

- **Timing is estimated by default.** The proportional aligner is smooth but not
  frame-accurate; switch to `--align ctc` for real forced alignment.
- **Layout/reading order is heuristic by default** — good for single/simple two-column
  born-digital PDFs; complex magazines/scans want `--layout doclayout` (or a reading-order
  model like PP-DocLayout / Surya).
- **OCR is off by default**; scanned PDFs need `--ocr surya|paddle`.
- **Table linearization** without a table-structure model reads in raw order; add Table
  Transformer / PP-Structure for true row/column reading.
- No speaker/voice-cloning, no per-word click-to-seek yet (easy to add: the viewer already
  has the word→time map).

See [`RESEARCH.md`](RESEARCH.md) for the full comparison behind every default.
