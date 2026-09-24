"""DocLayout-YOLO layout backend (RESEARCH.md §2, primary recommendation).

Runs a YOLOv10-based document layout detector on the rendered page image and converts the
detected regions into `Block`s (mapping pixel boxes back to PDF points). Fast enough for
CPU. Weights: `juliozhao/DocLayout-YOLO-DocStructBench` (repo: opendatalab/DocLayout-YOLO).

Install: pip install doclayout-yolo huggingface_hub   (in requirements-ml.txt)
Enable:  --layout doclayout
"""

from __future__ import annotations

from typing import List

from ..config import Config
from ..models import Block, Page

# DocLayout-YOLO class names -> our block kinds.
_CLASS_MAP = {
    "title": "heading",
    "plain text": "text",
    "figure": "figure",
    "figure_caption": "caption",
    "table": "table",
    "table_caption": "caption",
    "table_footnote": "caption",
    "isolate_formula": "text",
    "formula_caption": "caption",
    "abandon": "footer",  # headers/footers/page numbers
}


class DocLayoutYOLO:
    def __init__(self, config: Config) -> None:
        self.config = config
        try:
            from doclayout_yolo import YOLOv10  # noqa: WPS433 (optional heavy dep)
            from huggingface_hub import hf_hub_download  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - optional
            raise RuntimeError(
                "DocLayout-YOLO not installed. `pip install -r requirements-ml.txt`, "
                "or use layout_backend='heuristic'."
            ) from exc
        weights = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        self._model = YOLOv10(weights)

    def analyze(self, page: Page) -> List[Block]:
        if not page.image_path:
            raise RuntimeError(
                "DocLayout backend needs a rendered page image; enable page rendering."
            )
        # Scale factor from image pixels back to PDF points.
        # (render_dpi/72 was applied when rasterizing; invert it.)
        scale = 72.0 / self.config.render_dpi

        result = self._model.predict(page.image_path, imgsz=1024, conf=0.2)[0]
        blocks: List[Block] = []
        names = result.names
        for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
            kind = _CLASS_MAP.get(names[int(cls)], "text")
            bbox = tuple(c * scale for c in box)  # -> PDF points
            # Attach the born-digital words that fall inside this region (keeps highlight
            # geometry exact even when the layout model supplies the region).
            words = [w for w in page.words if _center_in(w.bbox, bbox)]
            text = " ".join(w.text for w in words)
            blocks.append(Block(bbox=bbox, page=page.number, kind=kind, text=text, words=words))

        # Reading order: reuse the heuristic orderer.
        from ..layout import order_blocks

        blocks = order_blocks(blocks, page)
        page.blocks = blocks
        return blocks


def _center_in(inner, outer) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]
