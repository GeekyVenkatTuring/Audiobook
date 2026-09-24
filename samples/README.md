# Samples

We deliberately do **not** commit large binary PDFs to the repo. Instead, a tiny
sample PDF is generated on demand so you can exercise the whole pipeline.

## Generate a sample PDF

```bash
python -c "from backend.utils.pdfgen import make_sample_pdf; make_sample_pdf('samples/sample.pdf')"
```

This produces a one-page PDF containing a heading, two body paragraphs, a vector
diagram (detected as a *figure* and skipped by default), a figure caption, and a
small table (detected and skipped by default, or linearized when
`read_tables=true`). `samples/*.pdf` is git-ignored.

## Try it end-to-end (no server)

```bash
python -m backend.smoke
```

## Try it through the API

```bash
uvicorn backend.main:app --reload
# then open http://127.0.0.1:8000 and upload samples/sample.pdf
# or:
curl -F "file=@samples/sample.pdf" -F "read_tables=true" \
     http://127.0.0.1:8000/api/process | python -m json.tool
```

## Testing scanned/OCR pages

To test the OCR path you need a scanned/image-only PDF (a PDF whose pages are
just images with no text layer). Provide your own, and either install the
`tesseract` system binary (default fallback) or the ML extras and pass
`ocr_provider=got_ocr` / `doctr`.
