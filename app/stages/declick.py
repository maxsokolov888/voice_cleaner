"""Короткие высокочастотные щелчки заменяются интерполяцией соседних сэмплов."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

SR_DEFAULT = 48_000
MAX_CLICK_S = 0.008
HALF_WIDTH_S = 0.002
CHUNK_S = 60
CONTEXT_S = 1


def declick_file(path: Path, sr: int, on_fraction=None) -> list[dict]:
    """Правит WAV на месте. Возвращает найденные щелчки в сэмплах файла."""
    repairs: list[dict] = []
    hop = CHUNK_S * sr
    context = CONTEXT_S * sr
    with sf.SoundFile(path, mode="r+") as audio:
        total = len(audio)
        pos = 0
        while pos < total:
            end = min(total, pos + hop)
            read_start = max(0, pos - context)
            read_end = min(total, end + context)
            audio.seek(read_start)
            block = audio.read(read_end - read_start, dtype="float32", always_2d=True)
            core = pos - read_start
            core_end = core + (end - pos)
            fixed, found = declick_block(block, sr)
            write_a = None
            write_b = None
            for item in found:
                if item["end"] <= core or item["start"] >= core_end:
                    continue
                write_a = item["start"] if write_a is None else min(write_a, item["start"])
                write_b = item["end"] if write_b is None else max(write_b, item["end"])
                mid = (item["start"] + item["end"]) / 2
                if core <= mid < core_end:
                    saved = dict(item)
                    saved["start"] += read_start
                    saved["end"] += read_start
                    repairs.append(saved)
            if write_a is not None:
                audio.seek(read_start + write_a)
                audio.write(fixed[write_a:write_b])
            pos = end
            if on_fraction:
                on_fraction(pos / total)
    return repairs


def declick_block(block: np.ndarray, sr: int) -> tuple[np.ndarray, list[dict]]:
    out = np.array(block, dtype=np.float32, copy=True)
    found: list[dict] = []
    for channel in range(out.shape[1]):
        samples = out[:, channel]
        spans = _click_spans(samples, sr)
        for start, end, peak_before, peak_after in spans:
            found.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "channel": channel,
                    "peak_before": float(peak_before),
                    "peak_after": float(peak_after),
                }
            )
    return out, found


def _click_spans(samples: np.ndarray, sr: int) -> list[tuple[int, int, float, float]]:
    if len(samples) < sr // 50:
        return []
    sos = butter(4, 4000, btype="highpass", fs=sr, output="sos")
    high = sosfiltfilt(sos, samples).astype(np.float32)
    win = max(3, int(0.02 * sr) | 1)
    baseline = uniform_filter1d(np.abs(high), size=win)
    local = uniform_filter1d(np.abs(samples), size=win)
    mask = np.abs(high) > np.maximum(baseline * 8.0, 1e-3)
    max_len = int(MAX_CLICK_S * sr)
    half_max = max(2, int(HALF_WIDTH_S * sr))
    spans: list[tuple[int, int, float, float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        if j - i <= max_len:
            seg = samples[i:j]
            peak = float(np.max(np.abs(seg)))
            high_energy = float(np.sum(high[i:j] ** 2))
            full_energy = float(np.sum(seg**2)) + 1e-12
            width = int(np.sum(np.abs(seg) >= peak * 0.5))
            peak_idx = i + int(np.argmax(np.abs(seg)))
            local_level = float(local[peak_idx])
            sticks_out = peak >= max(local_level * 8.0, 0.02)
            if sticks_out and peak >= 0.02 and width <= half_max and high_energy > 0.3 * full_energy:
                _interpolate(samples, i, j)
                peak_after = float(np.max(np.abs(samples[i:j]))) if j > i else 0.0
                spans.append((i, j, peak, peak_after))
        i = j
    return spans


def _interpolate(samples: np.ndarray, start: int, end: int) -> None:
    left = samples[start - 1] if start > 0 else 0.0
    right = samples[end] if end < len(samples) else left
    length = end - start
    if length <= 0:
        return
    samples[start:end] = np.linspace(left, right, length, dtype=np.float32)
