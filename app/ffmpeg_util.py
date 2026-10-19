"""Вызовы ffmpeg и ffprobe."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    return env


def tool(name: str) -> str:
    found = shutil.which(name, path=_env()["PATH"])
    if found:
        return found
    candidate = Path("/opt/homebrew/bin") / name
    if candidate.exists():
        return str(candidate)
    raise RuntimeError(f"Не найден {name}. Установите его через Homebrew.")


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        env=_env(),
    )


def duration_seconds(path: Path) -> float:
    result = run(
        [
            tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe не прочитал длительность")
    return float(result.stdout.strip())


def loudnorm_stats(path: Path) -> dict[str, str]:
    result = run(
        [
            tool("ffmpeg"),
            "-i",
            str(path),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    text = result.stderr
    start = text.rfind("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("ffmpeg не вернул замер громкости")
    data = json.loads(text[start : end + 1])
    needed = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if any(data.get(key) in (None, "inf", "-inf") for key in needed):
        raise RuntimeError("В записи слишком мало звука для нормализации громкости")
    return data
