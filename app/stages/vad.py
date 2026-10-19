"""Границы речи через Silero VAD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

_MODEL = None


def load_vad():
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
    return _MODEL


def speech_regions(path: Path, sr: int) -> list[tuple[int, int]]:
    if sr % 16_000 != 0:
        raise RuntimeError("VAD ждёт частоту, кратную 16 кГц")
    ratio = sr // 16_000
    model = load_vad()
    with sf.SoundFile(path) as audio:
        total = len(audio)
        low = np.empty(total // ratio, dtype=np.float32)
        offset = 0
        while True:
            block = audio.read(sr * 60, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            mono = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
            if len(mono) % ratio:
                mono = np.pad(mono, (0, ratio - len(mono) % ratio))
            piece = resample_poly(mono, 1, ratio).astype(np.float32)
            take = min(len(piece), len(low) - offset)
            if take <= 0:
                break
            low[offset : offset + take] = piece[:take]
            offset += take
    wav = torch.from_numpy(low[:offset])
    from silero_vad import get_speech_timestamps

    stamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16_000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
        return_seconds=False,
    )
    regions = []
    for stamp in stamps:
        start = int(stamp["start"]) * ratio
        end = min(total, int(stamp["end"]) * ratio)
        if end > start:
            regions.append((start, end))
    return regions
