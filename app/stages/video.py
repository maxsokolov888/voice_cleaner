"""Видео с очищенным голосом, не тяжелее лимита Telegram."""

from __future__ import annotations

import json
from pathlib import Path

from app.ffmpeg_util import run, tool
from app.pipeline import process

TARGET_BYTES = 48 * 1024 * 1024
MEDIA_BUDGET = int(TARGET_BYTES * 0.96)
AUDIO_BPS = 96_000
CRF = "20"
EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def video_bitrate(duration: float) -> int:
    if duration <= 0:
        raise RuntimeError("У видео нет длительности")
    return max(40_000, int(MEDIA_BUDGET * 8 / duration - AUDIO_BPS))


def frame_size(width: int, height: int, duration: float) -> tuple[int, int]:
    return _even_pair(width, height, _height_cap(video_bitrate(duration)))


def process_video(src: Path, dst: Path, workdir: Path, progress=None) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)

    def report(stage: str, percent: float) -> None:
        if progress:
            progress(stage, percent)

    report("Разбор видео", 2)
    info = probe_video(src)
    extracted = workdir / "input-audio.m4a"
    report("Извлечение звука", 4)
    extract_audio(src, extracted)
    voice = workdir / "voice.mp3"

    def audio_progress(stage: str, percent: float) -> None:
        if stage == "Готово":
            stage = "Звук готов"
        report(stage, 5 + percent * 0.78)

    summary = process(extracted, voice, workdir, audio_progress)
    report("Сжатие видео", 86)
    encode_telegram(src, voice, dst, info)
    voice.unlink(missing_ok=True)
    extracted.unlink(missing_ok=True)
    summary["video_bytes"] = dst.stat().st_size
    summary["video_width"] = info["out_width"]
    summary["video_height"] = info["out_height"]
    report("Готово", 100)
    return summary


def probe_video(path: Path) -> dict:
    result = run(
        [
            tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=codec_type,width,height,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "ffprobe не открыл видео")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if video is None:
        raise RuntimeError("В файле нет видео")
    if audio is None:
        raise RuntimeError("В видео нет звуковой дорожки")
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width < 2 or height < 2:
        raise RuntimeError("Не удалось прочитать размер кадра")
    duration = _positive(video.get("duration")) or _positive((data.get("format") or {}).get("duration"))
    if duration is None:
        raise RuntimeError("У видео нет длительности")
    bitrate = video_bitrate(duration)
    out_width, out_height = frame_size(width, height, duration)
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "bitrate": bitrate,
        "out_width": out_width,
        "out_height": out_height,
    }


def extract_audio(src: Path, dst: Path) -> None:
    _ffmpeg(
        [
            "-i",
            str(src),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(dst),
        ],
        "Не удалось извлечь звук из видео",
    )


def encode_telegram(src: Path, audio: Path, dst: Path, info: dict) -> None:
    draft = dst.with_name(dst.stem + "-draft.mp4")
    _encode_crf(src, audio, draft, info)
    if draft.stat().st_size <= TARGET_BYTES:
        draft.replace(dst)
        return
    bitrate = info["bitrate"]
    for factor in (1.0, 0.82, 0.65):
        _encode_two_pass(src, audio, draft, info, max(40_000, int(bitrate * factor)))
        if draft.stat().st_size <= TARGET_BYTES:
            break
    draft.replace(dst)


def _encode_crf(src: Path, audio: Path, dst: Path, info: dict) -> None:
    _ffmpeg(
        _common_args(src, audio, info)
        + [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            CRF,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            "-t",
            f"{info['duration']:.3f}",
            str(dst),
        ],
        "Не удалось собрать видео",
    )


def _encode_two_pass(src: Path, audio: Path, dst: Path, info: dict, bitrate: int) -> None:
    log = dst.with_name("x264pass")
    rate = str(bitrate)
    duration = f"{info['duration']:.3f}"
    _ffmpeg(
        [
            "-i",
            str(src),
            "-an",
            "-vf",
            _scale(info),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-b:v",
            rate,
            "-maxrate",
            rate,
            "-bufsize",
            str(bitrate * 2),
            "-pix_fmt",
            "yuv420p",
            "-pass",
            "1",
            "-passlogfile",
            str(log),
            "-t",
            duration,
            "-f",
            "null",
            "-",
        ],
        "Не удалось сжать видео",
    )
    _ffmpeg(
        _common_args(src, audio, info)
        + [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-b:v",
            rate,
            "-maxrate",
            rate,
            "-bufsize",
            str(bitrate * 2),
            "-pix_fmt",
            "yuv420p",
            "-pass",
            "2",
            "-passlogfile",
            str(log),
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            "-t",
            duration,
            str(dst),
        ],
        "Не удалось сжать видео",
    )
    for extra in log.parent.glob(log.name + "*"):
        extra.unlink(missing_ok=True)


def _common_args(src: Path, audio: Path, info: dict) -> list[str]:
    duration = f"{info['duration']:.3f}"
    return [
        "-i",
        str(src),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        _scale(info),
        "-af",
        f"apad=whole_dur={duration},atrim=end={duration}",
    ]


def _scale(info: dict) -> str:
    return f"scale={info['out_width']}:{info['out_height']}:flags=lanczos"


def _height_cap(video_bps: int) -> int:
    if video_bps >= 1_200_000:
        return 1080
    if video_bps >= 600_000:
        return 720
    return 480


def _even_pair(width: int, height: int, cap: int) -> tuple[int, int]:
    if height > cap:
        width = int(round(width * cap / height))
        height = cap
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    return max(2, width), max(2, height)


def _positive(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _ffmpeg(args: list[str], failure: str) -> None:
    result = run([tool("ffmpeg"), "-y", *args])
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        tail = detail[-1] if detail else failure
        raise RuntimeError(tail)
