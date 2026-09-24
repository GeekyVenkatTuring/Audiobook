"""Optional Hugging Face OCR providers (neural, heavy).

These are OFF by default and require `pip install -r requirements-ml.txt`.
They are provided as working stubs wired to the transformers API so that
enabling them is a one-liner once the weights are available.

Primary recommendation: GOT-OCR2.0 (stepfun-ai/GOT-OCR2_0). It is a compact
(~580M) end-to-end OCR-2.0 model that outputs clean reading-ordered text
(and can emit structured/formatted output), with a permissive license and a
transformers-native mirror (yonigozlan/GOT-OCR-2.0-hf). See docs/RESEARCH.md.

Lightweight neural fallback: docTR (detection + recognition, gives word boxes).
"""
from __future__ import annotations

from typing import List

from .base import OCRLine, OCRProvider


class GOTOCRProvider(OCRProvider):
    """GOT-OCR2.0 via transformers.

    Note: GOT-OCR2.0 returns document *text* (optionally formatted), not
    per-line bounding boxes. We therefore return a single full-page line whose
    bbox spans the page; highlighting for such pages is page-level. For box-level
    highlighting on scanned pages, prefer the Tesseract or docTR providers.
    """

    name = "got_ocr"

    def __init__(self, model_id: str = "stepfun-ai/GOT-OCR2_0") -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except Exception:
            return False

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def recognize(self, image_png: bytes) -> List[OCRLine]:
        import io

        from PIL import Image

        self._load()
        img = Image.open(io.BytesIO(image_png)).convert("RGB")
        inputs = self._processor(img, return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=4096)
        text = self._processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        w, h = img.size
        return [(text, (0.0, 0.0, float(w), float(h)))] if text else []


class DocTROCRProvider(OCRProvider):
    """docTR OCR (word-level boxes). Lighter than GOT-OCR2.0."""

    name = "doctr"

    def __init__(self) -> None:
        self._predictor = None

    def available(self) -> bool:
        try:
            import doctr  # noqa: F401

            return True
        except Exception:
            return False

    def _load(self) -> None:
        if self._predictor is not None:
            return
        from doctr.models import ocr_predictor

        self._predictor = ocr_predictor(pretrained=True)

    def recognize(self, image_png: bytes) -> List[OCRLine]:
        import io

        import numpy as np
        from PIL import Image

        self._load()
        img = np.array(Image.open(io.BytesIO(image_png)).convert("RGB"))
        H, W = img.shape[:2]
        result = self._predictor([img])
        out: List[OCRLine] = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words]
                    if not words:
                        continue
                    (x0, y0), (x1, y1) = line.geometry  # relative coords
                    out.append(
                        (
                            " ".join(words),
                            (x0 * W, y0 * H, x1 * W, y1 * H),
                        )
                    )
        return out
