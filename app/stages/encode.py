"""Двухпроходная громкость и mp3 44,1 кГц, стерео, 128 кбит/с."""

from __future__ import annotations

from pathlib import Path

from app.ffmpeg_util import loudnorm_stats, run, tool


def encode_mp3(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        stats = loudnorm_stats(src)
        audio_filter = (
            "loudnorm=I=-16:TP=-1.5:LRA=11:"
            f"measured_I={stats['input_i']}:"
            f"measured_TP={stats['input_tp']}:"
            f"measured_LRA={stats['input_lra']}:"
            f"measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}:"
            "linear=true"
        )
    except RuntimeError:
        audio_filter = "alimiter=limit=0.89"
    result = run(
        [
            tool("ffmpeg"),
            "-y",
            "-i",
            str(src),
            "-af",
            audio_filter,
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(dst),
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        tail = detail[-1] if detail else "ffmpeg не собрал mp3"
        raise RuntimeError(tail)
