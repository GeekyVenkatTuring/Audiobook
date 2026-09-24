"""Text-to-speech backends (RESEARCH.md §4).

All engines implement the `TTSEngine` protocol: given a list of sentence strings and an
output directory, synthesize one WAV per sentence and return `TTSResult`s carrying the
clip path and its exact duration (which alignment uses). The default `estimated` engine
produces silent WAVs sized to an estimated speaking duration — deterministic and fully
offline, so the pipeline and timing JSON can be produced and verified with no downloads.
"""

from .base import TTSEngine, TTSResult
from .estimated import EstimatedTTS


def get_tts_engine(config):
    """Factory: return the configured TTS backend."""
    backend = config.tts_backend
    if backend == "estimated":
        return EstimatedTTS(config)
    if backend == "pyttsx3":
        from .pyttsx3_engine import Pyttsx3TTS

        return Pyttsx3TTS(config)
    if backend == "kokoro":
        from .kokoro_engine import KokoroTTS

        return KokoroTTS(config)
    if backend == "piper":
        from .piper_engine import PiperTTS

        return PiperTTS(config)
    raise ValueError(f"Unknown tts_backend: {backend!r}")


__all__ = ["TTSEngine", "TTSResult", "EstimatedTTS", "get_tts_engine"]
