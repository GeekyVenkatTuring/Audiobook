"""WAV helpers built on the Python stdlib `wave` module (no heavy deps).

All WAV providers in this project emit 16-bit PCM mono so that segments can be
concatenated losslessly and durations measured exactly.
"""
from __future__ import annotations

import wave
from typing import List

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2  # bytes (16-bit)
CHANNELS = 1


def write_silence(path: str, seconds: float, sample_rate: int = SAMPLE_RATE) -> float:
    """Write a mono 16-bit silent WAV of the given duration. Returns duration."""
    seconds = max(0.0, seconds)
    n_frames = int(round(seconds * sample_rate))
    with wave.open(path, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_frames)
    return n_frames / float(sample_rate)


def wav_duration(path: str) -> float:
    """Exact duration of a WAV file in seconds."""
    with wave.open(path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate() or SAMPLE_RATE
        return frames / float(rate)


def concat_wavs(paths: List[str], out_path: str) -> float:
    """Concatenate WAV files into one. Uses the first file's params; skips any
    file whose params differ (with the caller responsible for consistency).
    Returns the total duration in seconds."""
    if not paths:
        # produce a valid empty wav
        with wave.open(out_path, "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(SAMPLE_WIDTH)
            w.setframerate(SAMPLE_RATE)
        return 0.0

    with wave.open(paths[0], "rb") as first:
        params = first.getparams()

    total_frames = 0
    with wave.open(out_path, "wb") as out:
        out.setparams(params)
        for p in paths:
            with wave.open(p, "rb") as w:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                ):
                    # Skip incompatible file rather than corrupt the stream.
                    continue
                frames = w.readframes(w.getnframes())
                out.writeframes(frames)
                total_frames += w.getnframes()
    return total_frames / float(params.framerate)


def concat_bytes(paths: List[str], out_path: str) -> None:
    """Naive byte concatenation (used for MP3 segments). Playable for CBR MP3."""
    with open(out_path, "wb") as out:
        for p in paths:
            with open(p, "rb") as f:
                out.write(f.read())
