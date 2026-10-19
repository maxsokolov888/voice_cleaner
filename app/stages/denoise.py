"""Изоляция речи DeepFilterNet3 кусками около минуты."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.paths import ROOT

CHUNK_S = 60
OVERLAP_S = 0.5

_MODEL = None
_STATE = None


def _patch_torchaudio() -> None:
    """Новый torchaudio убрал backend.common, а DeepFilterNet всё ещё его импортирует."""
    _quiet_git()
    try:
        from torchaudio.backend.common import AudioMetaData  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(
            self,
            sample_rate: int = 0,
            num_frames: int = 0,
            num_channels: int = 0,
            bits_per_sample: int = 0,
            encoding: str = "",
        ) -> None:
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    common.AudioMetaData = AudioMetaData
    backend = types.ModuleType("torchaudio.backend")
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def _quiet_git() -> None:
    import subprocess

    if getattr(subprocess, "_auphonic_quiet_git", False):
        return
    original = subprocess.check_output

    def check_output(args, *pos, **kwargs):
        if isinstance(args, (list, tuple)) and args and args[0] == "git":
            kwargs["stderr"] = subprocess.DEVNULL
        return original(args, *pos, **kwargs)

    subprocess.check_output = check_output
    subprocess._auphonic_quiet_git = True


def load_model():
    global _MODEL, _STATE
    if _MODEL is not None:
        return _MODEL, _STATE
    _patch_torchaudio()
    from df.enhance import init_df

    log_dir = ROOT / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    model, state, _ = init_df(
        log_level="ERROR",
        log_file=str(log_dir / "df.log"),
    )
    model = model.to("cpu").eval()
    _MODEL = model
    _STATE = state
    return model, state


def denoise_file(src: Path, dst: Path, sr: int, on_fraction=None) -> None:
    model, state = load_model()
    chunk = CHUNK_S * sr
    overlap = int(OVERLAP_S * sr)
    hop = chunk - overlap
    with sf.SoundFile(src) as fin, sf.SoundFile(
        dst,
        "w",
        samplerate=sr,
        channels=fin.channels,
        subtype="FLOAT",
    ) as fout:
        total = len(fin)
        pos = 0
        tail = None
        while pos < total:
            take = min(chunk, total - pos)
            fin.seek(pos)
            block = fin.read(take, dtype="float32", always_2d=True)
            enhanced = np.column_stack(
                [_denoise_channel(block[:, ch], model, state) for ch in range(block.shape[1])]
            )
            if len(enhanced) != take:
                enhanced = _fit(enhanced, take)
            if tail is None:
                if pos + take >= total:
                    fout.write(enhanced)
                else:
                    fout.write(enhanced[:-overlap])
                    tail = enhanced[-overlap:].copy()
            else:
                ov = min(overlap, len(enhanced), len(tail))
                fade = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
                mixed = tail[:ov] * (1.0 - fade) + enhanced[:ov] * fade
                fout.write(mixed)
                rest = enhanced[ov:]
                if pos + take >= total:
                    fout.write(rest)
                    tail = None
                else:
                    fout.write(rest[:-overlap])
                    tail = rest[-overlap:].copy()
            if pos + take >= total:
                break
            pos += hop
            if on_fraction:
                on_fraction(min(1.0, pos / total))
        if fout.frames != total:
            raise RuntimeError("Шумодав изменил длину файла")


def _denoise_channel(samples: np.ndarray, model, state) -> np.ndarray:
    from df.enhance import enhance

    tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        enhanced = enhance(model, state, tensor, pad=True)
    out = enhanced.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return _fit(out, len(samples))


def _fit(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.ndim == 1:
        if len(audio) >= length:
            return audio[:length]
        return np.pad(audio, (0, length - len(audio)))
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, ((0, length - len(audio)), (0, 0)))
