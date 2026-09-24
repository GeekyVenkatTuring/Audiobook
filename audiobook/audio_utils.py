"""Small WAV helpers built on the standard library only (no numpy required).

The pipeline concatenates per-sentence audio clips and needs to know each clip's exact
duration for alignment. These helpers read/write 16-bit mono PCM WAV files.
"""

from __future__ import annotations

import math
import struct
import wave
from typing import List


def write_silence(path: str, duration_s: float, sample_rate: int) -> None:
    """Write a mono 16-bit silent WAV of the given duration (used by 'estimated' TTS)."""
    n = int(round(duration_s * sample_rate))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)


def write_tone(path: str, duration_s: float, sample_rate: int, freq: float = 220.0,
               amplitude: float = 0.15) -> None:
    """Write a quiet sine tone (used only for audible offline smoke tests, optional)."""
    n = int(round(duration_s * sample_rate))
    frames = bytearray()
    for i in range(n):
        val = int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", val)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))


def wav_duration(path: str) -> float:
    """Return the duration in seconds of a WAV file."""
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def concat_wavs(paths: List[str], out_path: str) -> float:
    """Concatenate mono WAVs (same rate/width) into one file. Returns total duration.

    If clips disagree on sample rate the first file's parameters win and others are assumed
    compatible (all pipeline-generated clips share `config.sample_rate`).
    """
    if not paths:
        raise ValueError("no wav clips to concatenate")
    with wave.open(paths[0], "rb") as first:
        params = first.getparams()
    total_frames = 0
    with wave.open(out_path, "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        for p in paths:
            with wave.open(p, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                out.writeframes(frames)
                total_frames += wf.getnframes()
    return total_frames / float(params.framerate)
