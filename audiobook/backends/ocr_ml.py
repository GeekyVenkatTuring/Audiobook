"""OCR backends for scanned pages (RESEARCH.md §3): Surya (default) or PaddleOCR.

Both produce recognised text lines/words with bounding boxes on the rendered page image;
this adapter converts them to `Word`s in PDF points so scanned pages feed the exact same
highlight data model as born-digital pages. Heavy deps are imported lazily.

Install: pip install -r requirements-ml.txt   (surya-ocr and/or paddleocr)
Enable:  --ocr surya   (or  --ocr paddle)
"""

from __future__ import annotations

from typing import List

from ..config import Config
from ..models import Page, Word


class MLDocumentOCR:
    def __init__(self, config: Config, backend: str) -> None:
        self.config = config
        self.backend = backend
        self._scale = 72.0 / config.render_dpi  # image px -> PDF points
        if backend == "surya":
            self._init_surya()
        elif backend == "paddle":
            self._init_paddle()
        else:  # pragma: no cover
            raise ValueError(backend)

    # -- Surya OCR (datalab-to/surya): widest multilingual coverage --
    def _init_surya(self) -> None:
        try:
            from surya.recognition import RecognitionPredictor  # noqa: WPS433
            from surya.detection import DetectionPredictor  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "surya-ocr not installed. `pip install surya-ocr` (see requirements-ml.txt)."
            ) from exc
        self._det = DetectionPredictor()
        self._rec = RecognitionPredictor()

    # -- PaddleOCR: Apache-2.0, 100+ languages --
    def _init_paddle(self) -> None:
        try:
            from paddleocr import PaddleOCR  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "paddleocr not installed. `pip install paddleocr` (see requirements-ml.txt)."
            ) from exc
        self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def recognize(self, page: Page, image_path: str) -> List[Word]:
        from PIL import Image  # noqa: WPS433 (dependency of both OCR stacks)

        if self.backend == "surya":
            image = Image.open(image_path)
            preds = self._rec([image], det_predictor=self._det)[0]
            return self._words_from_surya(preds, page)
        # paddle
        result = self._paddle.ocr(image_path, cls=True)
        return self._words_from_paddle(result, page)

    def _words_from_surya(self, preds, page: Page) -> List[Word]:
        words: List[Word] = []
        for line in getattr(preds, "text_lines", []):
            # Split a recognised line into words, spreading the line bbox across them.
            tokens = line.text.split()
            if not tokens:
                continue
            x0, y0, x1, y1 = [c * self._scale for c in line.bbox]
            words.extend(_spread_line(tokens, x0, y0, x1, y1, page.number))
        return words

    def _words_from_paddle(self, result, page: Page) -> List[Word]:
        words: List[Word] = []
        for page_res in result or []:
            for box, (text, _conf) in page_res or []:
                xs = [p[0] * self._scale for p in box]
                ys = [p[1] * self._scale for p in box]
                tokens = text.split()
                if tokens:
                    words.extend(
                        _spread_line(tokens, min(xs), min(ys), max(xs), max(ys), page.number)
                    )
        return words


def _spread_line(tokens, x0, y0, x1, y1, page_no) -> List[Word]:
    """Distribute a line bbox across its tokens proportional to token length."""
    weights = [max(len(t), 1) for t in tokens]
    total = sum(weights)
    width = x1 - x0
    out: List[Word] = []
    cx = x0
    for tok, w in zip(tokens, weights):
        tw = width * (w / total)
        out.append(Word(text=tok, bbox=(cx, y0, cx + tw, y1), page=page_no))
        cx += tw
    return out
