# PDF Audiobook

Turn any PDF into an audiobook with **synchronized text highlighting** — it reads
the words aloud and highlights the text on the page as it goes, Kindle
read-aloud style. It understands page **layout** (headings, paragraphs, figures,
tables, captions), **skips diagrams** that have no words, can **linearize
tables**, and falls back to **OCR** for scanned/image-only PDFs.

> Status: working reference implementation. The full pipeline runs out of the
> box with only light dependencies (using a silent "estimator" voice so timing +
> highlighting work everywhere); real speech and neural OCR are one-line opt-ins.

---

## What it does

1. **Ingest & analyze layout** with PyMuPDF — extract text as blocks/lines with
   bounding boxes, detect images and vector diagrams, and detect tables.
2. **Classify** each region: `heading | text | figure | table | caption`, and
   decide a reading order and what to **read vs skip**.
3. **OCR fallback** — pages with no text layer are rendered and OCR'd.
4. **Synthesize audio** for the readable segments and build a **timeline**
   (start/end time per segment) mapped to each segment's bounding boxes.
5. **Emit a manifest** (`manifest.json`) + concatenated **audio** + the original
   **PDF**. The web UI renders the PDF with pdf.js and highlights each segment's
   boxes as the audio plays; click a segment to seek.

---

## Key decisions (from `docs/RESEARCH.md`)

- **Layout handling:** PyMuPDF native structure is the default (fast, no model
  downloads): `get_text("dict")` for block/line/span boxes + font sizes,
  `get_drawings()` clustered into diagram regions, and `find_tables()` for
  tables. A deep-learning layout backend (**PP-StructureV3 / PP-DocLayout**,
  LayoutParser, table-transformer) is documented as an optional upgrade for
  scans and complex multi-column documents.
- **Diagrams:** **skipped from audio by default** (an image/chart has no words to
  read); their **captions are read**. Configurable via `read_figures`.
- **Tables:** **skipped by default** (linear reading of a grid is confusing);
  when `read_tables` is on they are **linearized row-by-row** with
  `header: value` pairs. Skipped items are announced ("Figure skipped.") so the
  listener stays oriented (toggle `speak_placeholders`).
- **OCR (recommended model): GOT-OCR2.0** (`stepfun-ai/GOT-OCR2_0`) — best
  accuracy + reading-order/structure preservation, permissive license, compact
  (~580M), transformers-native. **docTR** is the lightweight fallback that gives
  word-level boxes. A **Tesseract** provider ships as the always-available
  default so the core needs no multi-GB downloads. See the reasoning in
  `docs/RESEARCH.md`.
- **TTS:** an **estimator** (silent WAV sized to estimated durations) is the
  zero-dependency default so timing/highlighting always work; **pyttsx3** gives
  real offline speech when a system voice exists; **Kokoro-82M** is the
  high-quality optional extra. Word-level timing via Kokoro token timings or
  WhisperX/aeneas forced alignment is documented as an upgrade.

---

## Architecture

```
                +-------------------- FastAPI (backend/main.py) --------------------+
  PDF upload -> | /api/process                                                     |
                |    |                                                             |
                |    v                                                             |
                | pipeline.process_pdf                                             |
                |    |                                                             |
                |    +-- layout.analyze_document (PyMuPDF)                         |
                |    |      blocks/lines/boxes, get_drawings, find_tables          |
                |    |      classify + reading order; scanned? -> OCR provider     |
                |    |                                   (ocr/base.py + factory)   |
                |    +-- build segments (read vs skip, linearize tables)           |
                |    +-- TTS provider (tts/base.py + factory) -> per-seg audio     |
                |    +-- concat audio + write manifest.json                        |
                |                                                                  |
                | GET /api/jobs/{id}/{manifest,audio,pdf}   + static frontend      |
                +------------------------------------------------------------------+
                                         |
                     manifest.json + audio + pdf
                                         v
      frontend/ (pdf.js): render pages -> overlay highlight boxes ->
      audio.currentTime drives active segment highlight + auto-scroll; click to seek
```

- `backend/layout.py` — PyMuPDF layout analysis & classification.
- `backend/ocr/` — `base.py` interface, `tesseract_provider.py` (default),
  `hf_provider.py` (GOT-OCR2.0 / docTR, optional), `factory.py`.
- `backend/tts/` — `base.py` interface, `estimator_provider.py` (default),
  `pyttsx3_provider.py`, `gtts_provider.py`, `kokoro_provider.py`, `factory.py`.
- `backend/pipeline.py` — orchestration; `backend/manifest.py` — data models.
- `backend/main.py` — FastAPI endpoints + static frontend hosting.
- `frontend/` — `index.html`, `app.js`, `styles.css` (pdf.js via CDN).

### Manifest shape (what the frontend consumes)
```jsonc
{
  "version": 1,
  "pages": [{ "index": 0, "width": 595, "height": 842, "ocr": false }],
  "segments": [
    { "id": 0, "page": 0, "type": "heading", "text": "…",
      "bboxes": [[x0,y0,x1,y1]], "start": 0.0, "end": 1.8,
      "spoken": true, "placeholder": false }
  ],
  "audio_file": "audio.wav", "audio_duration": 21.0,
  "tts_provider": "estimator", "ocr_used": false, "notes": []
}
```
Bounding boxes are in **PDF points, top-left origin** (matching the pdf.js
viewport), so the frontend scales them by `renderScale` directly.

---

## Setup & run

```bash
git clone https://github.com/GeekyVenkatTuring/Audiobook.git
cd Audiobook

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # light, fast, no big downloads

# run the web app (serves API + frontend on http://127.0.0.1:8000)
uvicorn backend.main:app --reload
# open http://127.0.0.1:8000, upload a PDF, press play
```

Verify the pipeline without a browser:
```bash
python -m backend.smoke     # generates a tiny PDF and runs it end-to-end
pytest -q                   # unit tests
```

### Optional: real / high-quality providers
```bash
# real offline speech (needs a system voice engine):
#   Linux:  sudo apt-get install espeak-ng
#   macOS:  built-in ;  Windows: built-in SAPI5
# then choose tts_provider=pyttsx3 in the UI (or 'auto')

# OCR for scanned PDFs (default fallback):
#   Linux:  sudo apt-get install tesseract-ocr

# neural OCR + high-quality TTS + DL layout (multi-GB downloads):
pip install -r requirements-ml.txt
# then: ocr_provider=got_ocr | doctr , tts_provider=kokoro
```

### Options (form fields on `POST /api/process`)
| field | default | meaning |
|---|---|---|
| `read_figures` | false | read figure/diagram regions |
| `read_tables` | false | linearize + read tables row-by-row |
| `read_captions` | true | read figure/table captions |
| `speak_placeholders` | true | announce skipped figures/tables |
| `ocr_provider` | auto | `auto\|tesseract\|got_ocr\|doctr\|none` |
| `tts_provider` | auto | `auto\|estimator\|pyttsx3\|gtts\|kokoro` |
| `wpm` | 165 | words-per-minute for estimator/word timing |

---

## Limitations & roadmap
- **Reading order** for complex multi-column layouts is heuristic (two-column
  split); PP-DocLayoutV2's learned reading order is the upgrade path.
- **Highlighting** is line/paragraph-level. Word-level needs a timing-emitting
  TTS (Kokoro) or forced alignment (WhisperX/aeneas) — interface is ready.
- **Scanned-page highlighting** with GOT-OCR2.0 is page-level (it returns text,
  not boxes); use Tesseract/docTR for box-level highlights on scans.
- The **estimator** voice is silent by design (guarantees the pipeline runs
  anywhere); install a real TTS for actual narration.
- PDF rotation / non-zero MediaBox offsets are assumed absent (typical); can be
  handled by transforming boxes with the page matrix.
- Processing is **synchronous** per request; a job queue would suit large PDFs.

## License
MIT (see repository).
