"""m4a → WAV 48 кГц. Похожие стереоканалы сворачиваются в моно."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.ffmpeg_util import run, tool

SR = 48_000
CORR_THRESHOLD = 0.95


@dataclass
class AudioInfo:
    path: Path
    sr: int
    channels: int
    frames: int
    downmixed: bool


def decode_m4a(src: Path, dst: Path) -> AudioInfo:
    dst.parent.mkdir(parents=True, exist_ok=True)
    stereo = dst.with_name(dst.stem + "-stereo.wav")
    result = run(
        [
            tool("ffmpeg"),
            "-y",
            "-i",
            str(src),
            "-ar",
            str(SR),
            "-ac",
            "2",
            "-c:a",
            "pcm_f32le",
            str(stereo),
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        tail = detail[-1] if detail else "ffmpeg не смог открыть файл"
        raise RuntimeError(tail)

    downmixed = _collapse_if_similar(stereo, dst)
    if downmixed:
        stereo.unlink(missing_ok=True)
    elif dst != stereo:
        stereo.replace(dst)

    info = sf.info(dst)
    return AudioInfo(
        path=dst,
        sr=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        downmixed=downmixed,
    )


def _collapse_if_similar(stereo: Path, mono: Path) -> bool:
    with sf.SoundFile(stereo) as src:
        take = min(len(src), SR * 60)
        block = src.read(take, dtype="float32", always_2d=True)
        if block.shape[1] < 2:
            return False
        left = block[:, 0]
        right = block[:, 1]
        if np.std(left) < 1e-6 or np.std(right) < 1e-6:
            similar = True
        else:
            similar = float(np.corrcoef(left, right)[0, 1]) >= CORR_THRESHOLD
        if not similar:
            return False
        _write_mono(src, mono)
    return True


def _write_mono(src: sf.SoundFile, mono: Path) -> None:
    src.seek(0)
    with sf.SoundFile(
        mono,
        "w",
        samplerate=src.samplerate,
        channels=1,
        subtype="FLOAT",
    ) as out:
        while True:
            block = src.read(SR * 60, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            out.write(block.mean(axis=1))
