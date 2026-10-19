"""Плавное усиление участков без изменения числа сэмплов."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def apply_gains(path: Path, sr: int, regions: list[dict]) -> None:
    if not regions:
        return
    with sf.SoundFile(path, mode="r+") as audio:
        total = len(audio)
        while True:
            start = audio.tell()
            block = audio.read(sr * 60, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            end = start + len(block)
            env = np.ones(len(block), dtype=np.float32)
            for region in regions:
                region_start = int(region["start"])
                region_end = int(region["end"])
                a = max(start, region_start)
                b = min(end, region_end)
                if b <= a:
                    continue
                full = _envelope(region_end - region_start, sr, float(region["gain"]), int(region["ramp_ms"]))
                env[a - start : b - start] *= full[a - region_start : b - region_start]
            audio.seek(start)
            audio.write(block * env[:, None])
        if audio.frames != total:
            raise RuntimeError("Заглушение изменило длину файла")


def _envelope(length: int, sr: int, gain: float, ramp_ms: int) -> np.ndarray:
    ramp = min(int(sr * ramp_ms / 1000), length // 2)
    env = np.full(length, gain, dtype=np.float32)
    if ramp <= 1:
        return env
    curve = (1.0 - np.cos(np.linspace(0.0, np.pi, ramp, dtype=np.float32))) / 2.0
    env[:ramp] = 1.0 + (gain - 1.0) * curve
    env[-ramp:] = 1.0 + (gain - 1.0) * curve[::-1]
    return env


def region_rms(path: Path, start: int, end: int) -> float:
    with sf.SoundFile(path) as audio:
        audio.seek(max(0, start))
        block = audio.read(max(0, end - start), dtype="float32", always_2d=True)
    if len(block) == 0:
        return 0.0
    mono = block.mean(axis=1)
    return float(np.sqrt(np.mean(mono**2)))
