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
                full = _envelope(
                    region_end - region_start,
                    sr,
                    float(region["gain"]),
                    int(region["ramp_ms"]),
                    int(region.get("ramp_in_ms", region["ramp_ms"])),
                )
                env[a - start : b - start] *= full[a - region_start : b - region_start]
            audio.seek(start)
            audio.write(block * env[:, None])
        if audio.frames != total:
            raise RuntimeError("Заглушение изменило длину файла")


def _envelope(length: int, sr: int, gain: float, ramp_ms: int, ramp_in_ms: int | None = None) -> np.ndarray:
    fade_out = int(sr * ramp_ms / 1000)
    fade_in = int(sr * (ramp_ms if ramp_in_ms is None else ramp_in_ms) / 1000)
    if fade_out + fade_in > length and fade_out + fade_in > 0:
        scale = length / (fade_out + fade_in)
        fade_out = int(fade_out * scale)
        fade_in = int(fade_in * scale)
    env = np.full(length, gain, dtype=np.float32)
    if fade_out > 1:
        curve = (1.0 - np.cos(np.linspace(0.0, np.pi, fade_out, dtype=np.float32))) / 2.0
        env[:fade_out] = 1.0 + (gain - 1.0) * curve
    if fade_in > 1:
        curve = (1.0 - np.cos(np.linspace(0.0, np.pi, fade_in, dtype=np.float32))) / 2.0
        env[-fade_in:] = 1.0 + (gain - 1.0) * curve[::-1]
    return env


def region_rms(path: Path, start: int, end: int) -> float:
    with sf.SoundFile(path) as audio:
        audio.seek(max(0, start))
        block = audio.read(max(0, end - start), dtype="float32", always_2d=True)
    if len(block) == 0:
        return 0.0
    mono = block.mean(axis=1)
    return float(np.sqrt(np.mean(mono**2)))
