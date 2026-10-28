"""Вздохи заглушаются до неслышимого уровня. Сэмплы не вырезаются."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

MIN_GAP_S = 0.02
FADE_OUT_MS = 180
FADE_IN_MS = 90
ONSET_MIN_S = 0.12
ONSET_MAX_S = 1.0
VOICE_AC = 0.42
# Пауза громче этого относительно речи похожа на пропущенную фразу, её не глушим.
GAP_SPEECH_RATIO = 0.55


def find_breaths(path: Path, sr: int, speech: list[tuple[int, int]]) -> tuple[list[dict], float]:
    speech_rms = _speech_level(path, sr, speech)
    if speech_rms < 1e-5:
        return [], speech_rms
    breaths: list[dict] = []
    for start, end in _gap_bounds(speech, _length(path)):
        if end - start < int(MIN_GAP_S * sr):
            continue
        block = _read(path, start, end)
        mono = block.mean(axis=1) if block.ndim == 2 else block
        rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
        if rms > speech_rms * GAP_SPEECH_RATIO and (end - start) > int(0.35 * sr):
            continue
        kept = _shrink_release(mono, sr, min_release_s=0.04)
        if kept is None:
            continue
        rel_start, rel_end = kept
        seg = mono[rel_start:rel_end]
        rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
        breaths.append(_region(start + rel_start, start + rel_end, rms, sr))
    for start, end in speech:
        block = _read(path, start, end)
        mono = block.mean(axis=1) if block.ndim == 2 else block
        for rel_start, rel_end in _glued_breaths(mono, sr):
            seg = mono[rel_start:rel_end]
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
            breaths.append(_region(start + rel_start, start + rel_end, rms, sr))
        for rel_start, rel_end in _valleys(mono, sr, speech_rms):
            kept = _shrink_release(mono[rel_start:rel_end], sr, min_release_s=0.03)
            if kept is None:
                continue
            seg_start, seg_end = kept
            seg = mono[rel_start + seg_start : rel_start + seg_end]
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
            breaths.append(_region(start + rel_start + seg_start, start + rel_start + seg_end, rms, sr))
    return _merge(breaths), float(speech_rms)


def _valleys(mono: np.ndarray, sr: int, speech_rms: float) -> list[tuple[int, int]]:
    """Тихий провал внутри фразы: вдох, который VAD не отделил от слов."""
    if len(mono) < int(0.5 * sr):
        return []
    frame = max(1, int(0.02 * sr))
    levels: list[float] = []
    positions: list[int] = []
    for pos in range(0, len(mono) - frame + 1, frame):
        piece = mono[pos : pos + frame]
        levels.append(float(np.sqrt(np.mean(piece**2))))
        positions.append(pos)
    low = speech_rms * 0.25
    high = speech_rms * 0.50
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(levels):
        if levels[index] >= low:
            index += 1
            continue
        end_index = index
        while end_index < len(levels) and levels[end_index] < high:
            end_index += 1
        start = positions[index]
        end = min(len(mono), positions[end_index - 1] + frame)
        duration = (end - start) / sr
        core = min(levels[index:end_index])
        if 0.20 <= duration <= 1.15 and core < speech_rms * 0.14:
            seg_rms = float(np.sqrt(np.mean(mono[start:end] ** 2)))
            left = mono[max(0, start - int(0.2 * sr)) : start]
            right = mono[end : min(len(mono), end + int(0.2 * sr))]
            side = 0.0
            if len(left):
                side = max(side, float(np.sqrt(np.mean(left**2))))
            if len(right):
                side = max(side, float(np.sqrt(np.mean(right**2))))
            if side > speech_rms * 0.45 and seg_rms < side * 0.20:
                spans.append((start, end))
        index = max(end_index, index + 1)
    return spans


def _shrink_release(mono: np.ndarray, sr: int, min_release_s: float) -> tuple[int, int] | None:
    """Не глушит спад слова и конечный согласный в начале участка."""
    frame = max(1, int(0.02 * sr))
    if len(mono) < frame * 4:
        return None
    acs: list[float] = []
    levels: list[float] = []
    for pos in range(0, len(mono) - frame + 1, frame):
        piece = mono[pos : pos + frame]
        acs.append(_periodicity(piece, sr))
        levels.append(float(np.sqrt(np.mean(piece**2))))
    scan = min(len(acs), max(1, int(round(0.28 * sr / frame))))
    last_voiced = None
    for index in range(scan):
        if acs[index] >= 0.38:
            last_voiced = index
    min_index = int(round(min_release_s * sr / frame))
    hang = max(1, int(round(0.02 * sr / frame)))
    if last_voiced is None:
        start_index = min_index
    else:
        start_index = max(min_index, last_voiced + hang)
    end_index = len(acs)
    median = float(np.median(levels)) if levels else 0.0
    while end_index > start_index + 3:
        tail_voiced = acs[end_index - 1] >= 0.38
        tail_rising = median > 0 and levels[end_index - 1] > median * 1.8
        if not tail_voiced and not tail_rising:
            break
        end_index -= 1
    start = min(len(mono), start_index * frame)
    end = min(len(mono), end_index * frame)
    if end - start < int(0.08 * sr):
        return None
    return start, end


def _region(start: int, end: int, rms: float, sr: int) -> dict:
    return {
        "start": int(start),
        "end": int(end),
        "gain": 0.0,
        "ramp_ms": FADE_OUT_MS,
        "ramp_in_ms": FADE_IN_MS,
        "rms_before": float(rms),
        "label": "breath",
    }


def _glued_breaths(mono: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Вдох, который VAD приклеил к фразе: глухой и тише следующего голоса."""
    frame = max(1, int(0.03 * sr))
    hop = max(1, int(0.01 * sr))
    if len(mono) < int(0.45 * sr) or len(mono) < frame:
        return []
    acs: list[float] = []
    positions: list[int] = []
    for pos in range(0, len(mono) - frame + 1, hop):
        acs.append(_periodicity(mono[pos : pos + frame], sr))
        positions.append(pos)
    head = _edge_run(mono, sr, acs, positions, frame, from_start=True)
    if head is None:
        return []
    return [head]


def _edge_run(mono, sr, acs, positions, frame, from_start: bool) -> tuple[int, int] | None:
    order = range(len(acs)) if from_start else range(len(acs) - 1, -1, -1)
    voiced = 0
    last_quiet = None
    saw_voice = False
    for index in order:
        if acs[index] >= VOICE_AC:
            voiced += 1
            if voiced >= 3:
                saw_voice = True
                break
        else:
            voiced = 0
            last_quiet = index
    if not saw_voice or last_quiet is None:
        return None
    if from_start:
        start, end = 0, min(len(mono), positions[last_quiet] + frame)
        follow = mono[end : min(len(mono), end + int(0.25 * sr))]
    else:
        start, end = positions[last_quiet], len(mono)
        follow = mono[max(0, start - int(0.25 * sr)) : start]
    duration = (end - start) / sr
    if not (ONSET_MIN_S <= duration <= ONSET_MAX_S) or len(follow) < int(0.08 * sr):
        return None
    edge_rms = float(np.sqrt(np.mean(mono[start:end] ** 2)))
    follow_rms = float(np.sqrt(np.mean(follow**2)))
    if follow_rms < 1e-4 or edge_rms >= follow_rms * 0.5:
        return None
    return start, end


def _periodicity(piece: np.ndarray, sr: int) -> float:
    samples = piece.astype(np.float32, copy=False)
    centered = samples - float(samples.mean())
    energy = float(np.dot(centered, centered))
    if energy < 1e-8:
        return 0.0
    size = 1 << (len(centered) * 2 - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), size)
    low = max(1, int(sr / 340))
    high = min(len(corr) - 1, int(sr / 85))
    if high <= low:
        return 0.0
    return float(corr[low:high].max() / corr[0])


def _merge(items: list[dict]) -> list[dict]:
    if not items:
        return []
    items = sorted(items, key=lambda item: (item["start"], item["end"]))
    merged = [dict(items[0])]
    for item in items[1:]:
        last = merged[-1]
        if item["start"] <= last["end"]:
            last["end"] = max(last["end"], item["end"])
            last["gain"] = min(last["gain"], item["gain"])
            last["rms_before"] = max(last["rms_before"], item["rms_before"])
            last["ramp_ms"] = max(last["ramp_ms"], item["ramp_ms"])
            last["ramp_in_ms"] = max(last.get("ramp_in_ms", 0), item.get("ramp_in_ms", 0))
        else:
            merged.append(dict(item))
    return merged


def _gap_bounds(speech: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    gaps = []
    cursor = 0
    for start, end in speech:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length:
        gaps.append((cursor, length))
    return gaps


def _speech_level(path: Path, sr: int, speech: list[tuple[int, int]]) -> float:
    if not speech:
        return 0.0
    levels = []
    for start, end in speech[:24]:
        take = min(end, start + sr)
        block = _read(path, start, take)
        mono = block.mean(axis=1) if block.ndim == 2 else block
        if len(mono) == 0:
            continue
        levels.append(float(np.sqrt(np.mean(mono**2))))
    if not levels:
        return 0.0
    return float(np.median(levels))


def _length(path: Path) -> int:
    info = sf.info(path)
    return int(info.frames)


def _read(path: Path, start: int, end: int) -> np.ndarray:
    with sf.SoundFile(path) as audio:
        audio.seek(max(0, start))
        return audio.read(max(0, end - start), dtype="float32", always_2d=True)
