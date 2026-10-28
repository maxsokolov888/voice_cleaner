"""Склейка шагов. Длина сигнала не меняется до самого mp3."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from app.ffmpeg_util import duration_seconds
from app.stages.breaths import find_breaths
from app.stages.decode import decode_m4a
from app.stages.declick import declick_file
from app.stages.denoise import denoise_file
from app.stages.encode import encode_mp3
from app.stages.events import detect_events
from app.stages.gain import apply_gains, region_rms
from app.stages.vad import speech_regions


def process(src: Path, dst: Path, workdir: Path, progress=None) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    wav = workdir / "audio.wav"
    cleaned = workdir / "cleaned.wav"

    def report(stage: str, percent: float) -> None:
        if progress:
            progress(stage, percent)

    report("Декодирование", 5)
    info = decode_m4a(src, wav)
    wav_duration = info.frames / info.sr

    report("Щелчки", 8)
    clicks = declick_file(wav, info.sr, on_fraction=lambda f: report("Щелчки", 8 + 10 * f))

    report("Поиск кашля и ударов", 20)
    events = detect_events(wav, info.sr, on_fraction=lambda f: report("Поиск кашля и ударов", 20 + 15 * f))

    report("Очистка речи", 36)
    denoise_file(wav, cleaned, info.sr, on_fraction=lambda f: report("Очистка речи", 36 + 38 * f))
    wav.unlink(missing_ok=True)

    report("Поиск речи", 76)
    speech = speech_regions(cleaned, info.sr)
    report("Вздохи", 84)
    breaths, speech_rms = find_breaths(cleaned, info.sr, speech)

    report("Заглушение", 90)
    apply_gains(cleaned, info.sr, [*breaths, *events])
    _measure_after(cleaned, breaths, events, speech)

    report("Сборка mp3", 94)
    encode_mp3(cleaned, dst)
    cleaned_frames = sf.info(cleaned).frames
    cleaned.unlink(missing_ok=True)
    if cleaned_frames != info.frames:
        raise RuntimeError("Число сэмплов после очистки не совпало с исходным")

    output_duration = duration_seconds(dst)
    summary = {
        "input_duration": duration_seconds(src),
        "wav_duration": wav_duration,
        "output_duration": output_duration,
        "downmixed": info.downmixed,
        "channels": info.channels,
        "clicks": len(clicks),
        "click_peaks": _click_peaks(clicks),
        "speech_regions": len(speech),
        "speech_rms": speech_rms,
        "breaths": breaths,
        "events": events,
    }
    (workdir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report("Готово", 100)
    return summary


def _measure_after(path: Path, breaths: list[dict], events: list[dict], speech: list[tuple[int, int]]) -> None:
    for item in breaths:
        pad = int(max(item["ramp_ms"], item.get("ramp_in_ms", item["ramp_ms"])) / 1000 * 48_000)
        length = item["end"] - item["start"]
        pad = min(pad, max(0, (length - 1) // 3))
        start = item["start"] + pad
        end = max(start + 1, item["end"] - pad)
        item["rms_after"] = region_rms(path, start, end)
    for item in events:
        pad = int(0.02 * 48_000)
        start = item["start"] + pad
        end = max(start + 1, item["end"] - pad)
        item["rms_after"] = region_rms(path, start, end)
    if speech:
        levels = [region_rms(path, start, min(end, start + 48_000)) for start, end in speech[:12]]
        speech_after = sorted(levels)[len(levels) // 2]
    else:
        speech_after = 0.0
    for item in breaths:
        item["speech_rms_after"] = speech_after
    for item in events:
        item["speech_rms_after"] = speech_after


def _click_peaks(clicks: list[dict]) -> dict:
    if not clicks:
        return {"count": 0}
    before = [item["peak_before"] for item in clicks[:50]]
    after = [item["peak_after"] for item in clicks[:50]]
    return {
        "count": len(clicks),
        "median_peak_before": float(sorted(before)[len(before) // 2]),
        "median_peak_after": float(sorted(after)[len(after) // 2]),
    }


def problems(summary: dict) -> list[str]:
    issues = []
    if abs(summary["output_duration"] - summary["wav_duration"]) > 0.05:
        issues.append(
            f"Длительность mp3 {summary['output_duration']:.3f} с против {summary['wav_duration']:.3f} с звука"
        )
    speech = summary.get("speech_rms") or 0.0
    peaks = summary.get("click_peaks") or {}
    before = peaks.get("median_peak_before")
    after = peaks.get("median_peak_after")
    if peaks.get("count") and before and after is not None and after > before * 0.5:
        issues.append("Щелчки остались слишком громкими")
    for breath in summary["breaths"]:
        after_rms = breath.get("rms_after", 0.0)
        before_rms = breath.get("rms_before", 0.0)
        if speech and after_rms > speech * 0.02 and before_rms > speech * 0.05:
            issues.append("Вздох остался слишком громким относительно речи")
    for event in summary["events"]:
        after_rms = event.get("rms_after", 0.0)
        limit = max(speech * 0.05, 1e-4)
        if after_rms > limit:
            issues.append(f"Событие «{event['label']}» не ушло в тишину")
    return issues


def main() -> None:
    import sys

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    work = dst.parent / f"{dst.stem}-work"
    summary = process(src, dst, work, progress=lambda stage, percent: print(f"{percent:.0f} {stage}", flush=True))
    issues = problems(summary)
    print(
        f"длительность {summary['output_duration']:.3f} с, "
        f"вздохов {len(summary['breaths'])}, событий {len(summary['events'])}, щелчков {summary['clicks']}"
    )
    if issues:
        print("\n".join(dict.fromkeys(issues)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
