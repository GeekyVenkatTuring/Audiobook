"""FastAPI application: upload a PDF, process it, serve manifest/audio/pdf,
and serve the static frontend."""
from __future__ import annotations

import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR, FRONTEND_DIR, ProcessOptions
from .pipeline import process_pdf

app = FastAPI(
    title="PDF Audiobook",
    description="Turn any PDF into an audiobook with synchronized highlighting.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _job_dir(job_id: str) -> str:
    return os.path.join(str(DATA_DIR), job_id)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    read_figures: str = Form(None),
    read_tables: str = Form(None),
    read_captions: str = Form(None),
    speak_placeholders: str = Form(None),
    ocr_provider: str = Form(None),
    tts_provider: str = Form(None),
    wpm: str = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    opts = ProcessOptions.from_form(
        {
            "read_figures": read_figures,
            "read_tables": read_tables,
            "read_captions": read_captions,
            "speak_placeholders": speak_placeholders,
            "ocr_provider": ocr_provider,
            "tts_provider": tts_provider,
            "wpm": wpm,
        }
    )

    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    pdf_path = os.path.join(job_dir, "source.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        manifest = process_pdf(pdf_path, opts, job_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    payload = manifest.to_dict()
    payload["job_id"] = job_id
    payload["manifest_url"] = f"/api/jobs/{job_id}/manifest"
    payload["audio_url"] = f"/api/jobs/{job_id}/audio"
    payload["pdf_url"] = f"/api/jobs/{job_id}/pdf"
    return JSONResponse(payload)


@app.get("/api/jobs/{job_id}/manifest")
def get_manifest(job_id: str):
    path = os.path.join(_job_dir(job_id), "manifest.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Unknown job.")
    return FileResponse(path, media_type="application/json")


@app.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: str):
    d = _job_dir(job_id)
    for name, mt in (("audio.wav", "audio/wav"), ("audio.mp3", "audio/mpeg")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return FileResponse(p, media_type=mt)
    raise HTTPException(status_code=404, detail="Audio not found.")


@app.get("/api/jobs/{job_id}/pdf")
def get_pdf(job_id: str):
    p = os.path.join(_job_dir(job_id), "source.pdf")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="PDF not found.")
    return FileResponse(p, media_type="application/pdf")


# Serve the static frontend at the root (mounted last so /api/* wins).
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
