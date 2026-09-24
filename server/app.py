"""FastAPI server: serve the viewer and generated assets, plus a process-upload endpoint.

Routes:
    GET  /                      -> the single-page PDF.js viewer
    POST /api/process           -> upload a PDF (multipart), run the pipeline, return job id
    GET  /api/jobs/{id}/timing.json
    GET  /api/jobs/{id}/audio.wav
    GET  /api/jobs/{id}/source.pdf
    GET  /api/jobs/{id}/pages/{name}
    GET  /out/...               -> serve a pre-generated output dir (via --out)

Run:
    uvicorn server.app:app --reload
    # or, to also expose a pre-generated dir:
    python -m server.app --out output
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from audiobook.config import Config
from audiobook.pipeline import process_pdf

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
JOBS_DIR = Path(os.environ.get("AUDIOBOOK_JOBS_DIR", tempfile.gettempdir())) / "audiobook_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PDF → Audiobook")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    tts: str = "estimated",
    align: str = "proportional",
    table_mode: str = "announce",
) -> JSONResponse:
    """Accept a PDF upload, run the pipeline, and return the job id + summary."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / "input.pdf"
    with open(pdf_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    config = Config(tts_backend=tts, alignment_backend=align, table_mode=table_mode)
    result = process_pdf(str(pdf_path), str(job_dir), config=config)

    return JSONResponse({
        "job_id": job_id,
        "num_words": result.num_words,
        "duration": result.duration_s,
        "timing_url": f"/api/jobs/{job_id}/timing.json",
        "audio_url": f"/api/jobs/{job_id}/audio.wav",
        "pdf_url": f"/api/jobs/{job_id}/source.pdf",
    })


@app.get("/api/jobs/{job_id}/{asset:path}")
def job_asset(job_id: str, asset: str) -> FileResponse:
    # Prevent path traversal; only serve files inside the job dir.
    job_dir = (JOBS_DIR / job_id).resolve()
    target = (job_dir / asset).resolve()
    if not str(target).startswith(str(job_dir)) or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target)


# Serve the static frontend assets (viewer.js, style.css, pdf.js worker copy if vendored).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _mount_prebuilt(out_dir: str) -> None:
    """Expose a pre-generated output directory at /out for quick local viewing."""
    path = Path(out_dir).resolve()
    if path.is_dir():
        app.mount("/out", StaticFiles(directory=path), name="out")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the audiobook web viewer.")
    parser.add_argument("--out", default=None,
                        help="pre-generated output dir to serve at /out (from the CLI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.out:
        _mount_prebuilt(args.out)
        print(f"Serving pre-generated output at http://{args.host}:{args.port}/?src=/out")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
