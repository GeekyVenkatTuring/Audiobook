"""audiobook — turn a PDF into a narrated audiobook with word-level highlight timing.

The package is organised as a small, composable pipeline:

    ingest  ->  layout  ->  content_filter  ->  (ocr)  ->  tts  ->  alignment  ->  outputs

Every heavy/ML stage is behind a small interface with a light, dependency-free default,
so the core pipeline runs end-to-end on born-digital PDFs with no model downloads. See
RESEARCH.md for the recommended production backends behind each interface.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
