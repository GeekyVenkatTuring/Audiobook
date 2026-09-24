"""Select a TTS provider from an options string, with graceful fallback."""
from __future__ import annotations

from .base import TTSProvider
from .estimator_provider import EstimatorTTSProvider


def get_tts_provider(name: str, wpm: int = 165) -> TTSProvider:
    """Return a TTS provider.

    'auto'      -> pyttsx3 if available, else estimator (always works).
    'estimator' -> silent WAV sized to estimated durations (zero deps).
    'pyttsx3'   -> offline system speech.
    'gtts'      -> online Google TTS (MP3).
    'kokoro'    -> Kokoro-82M neural TTS (requires requirements-ml.txt).
    Any unavailable choice degrades to the estimator.
    """
    name = (name or "auto").lower()

    if name == "estimator":
        return EstimatorTTSProvider(wpm)

    if name == "pyttsx3":
        from .pyttsx3_provider import Pyttsx3TTSProvider

        p = Pyttsx3TTSProvider(wpm)
        return p if p.available() else EstimatorTTSProvider(wpm)

    if name == "gtts":
        from .gtts_provider import GTTSTTSProvider

        p = GTTSTTSProvider(wpm)
        return p if p.available() else EstimatorTTSProvider(wpm)

    if name == "kokoro":
        from .kokoro_provider import KokoroTTSProvider

        p = KokoroTTSProvider(wpm)
        return p if p.available() else EstimatorTTSProvider(wpm)

    # auto: prefer real offline speech, else estimator.
    try:
        from .pyttsx3_provider import Pyttsx3TTSProvider

        p = Pyttsx3TTSProvider(wpm)
        if p.available():
            return p
    except Exception:
        pass
    return EstimatorTTSProvider(wpm)
