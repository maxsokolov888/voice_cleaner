"""Кашель, чих и удары ищутся YAMNet по сигналу до шумодава."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.paths import ROOT

SR_IN = 48_000
SR_MODEL = 16_000
WINDOW = 15_600
HOP = 8_000
CHUNK_S = 60

MUTE_NAMES = {
    "Cough",
    "Throat clearing",
    "Sneeze",
    "Snort",
    "Hiccup",
    "Bang",
    "Slam",
    "Thump, thud",
    "Crash",
    "Smash, crash",
    "Breaking",
    "Shatter",
    "Glass",
    "Clatter",
    "Tap",
    "Tick",
    "Clicking",
    "Clink",
    "Ping",
    "Slap, smack",
    "Whack, thwack",
    "Crushing",
    "Crackle",
    "Knock",
    "Cupboard open or close",
    "Drawer open or close",
    "Dishes, pots, and pans",
    "Coin (dropping)",
    "Finger snapping",
}

_INTERPRETER = None
_LABELS: list[str] | None = None
_MUTE_INDEX: list[int] | None = None
_SPEECH_INDEX = 0


def model_paths() -> tuple[Path, Path]:
    folder = ROOT / "models"
    return folder / "yamnet.tflite", folder / "yamnet_class_map.csv"


def ensure_yamnet_files() -> tuple[Path, Path]:
    model, labels = model_paths()
    folder = model.parent
    folder.mkdir(parents=True, exist_ok=True)
    if not model.exists():
        _download(
            "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite",
            model,
        )
    if not labels.exists():
        _download(
            "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
            labels,
        )
    return model, labels


def load_yamnet():
    global _INTERPRETER, _LABELS, _MUTE_INDEX, _SPEECH_INDEX
    if _INTERPRETER is not None:
        return _INTERPRETER
    model, labels_path = ensure_yamnet_files()
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    import csv

    with labels_path.open(encoding="utf-8", newline="") as handle:
        labels = [row["display_name"] for row in csv.DictReader(handle)]
    _LABELS = labels
    _MUTE_INDEX = [i for i, name in enumerate(labels) if name in MUTE_NAMES]
    _SPEECH_INDEX = labels.index("Speech") if "Speech" in labels else 0
    _INTERPRETER = interpreter
    return interpreter


def detect_events(path: Path, sr: int, on_fraction=None) -> list[dict]:
    if sr != SR_IN:
        raise RuntimeError("Детектор событий ждёт 48 кГц")
    interpreter = load_yamnet()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    score_index = _score_output(output_details)
    scale = _analysis_scale(path)
    hits: list[dict] = []
    pending = np.zeros(0, dtype=np.float32)
    origin = 0
    total = 0
    with sf.SoundFile(path) as audio:
        total = len(audio)
        done = 0
        while done < total:
            block = audio.read(CHUNK_S * sr, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            mono = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
            padded = mono
            if len(padded) % 3:
                padded = np.pad(padded, (0, 3 - len(padded) % 3))
            low = resample_poly(padded, 1, 3).astype(np.float32)
            stream = np.concatenate([pending, low])
            produced = 0
            while produced + WINDOW <= len(stream):
                window = stream[produced : produced + WINDOW]
                scores = _infer(interpreter, input_detail, output_details[score_index], window, scale)
                hit = _classify(scores)
                if hit is not None:
                    start_16 = origin + produced
                    start = start_16 * 3
                    end = min(total, (start_16 + WINDOW) * 3)
                    hits.append(
                        {
                            "start": int(start),
                            "end": int(end),
                            "label": hit[0],
                            "score": hit[1],
                            "gain": 0.0,
                            "ramp_ms": 15,
                        }
                    )
                produced += HOP
            pending = stream[produced:]
            origin += produced
            done += len(block)
            if on_fraction:
                on_fraction(done / total)
    if len(pending) >= int(0.3 * SR_MODEL):
        window = np.zeros(WINDOW, dtype=np.float32)
        window[: len(pending)] = pending
        scores = _infer(interpreter, input_detail, output_details[score_index], window, scale)
        hit = _classify(scores)
        if hit is not None:
            start = origin * 3
            end = min(total, (origin + len(pending)) * 3)
            hits.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "label": hit[0],
                    "score": hit[1],
                    "gain": 0.0,
                    "ramp_ms": 15,
                }
            )
    return _merge(hits)


def _analysis_scale(path: Path) -> float:
    peak = 0.0
    with sf.SoundFile(path) as audio:
        while True:
            block = audio.read(SR_IN * 60, dtype="float32")
            if len(block) == 0:
                break
            peak = max(peak, float(np.max(np.abs(block))))
    if peak < 1e-6:
        return 1.0
    return 0.9 / peak


def _score_output(details: list[dict]) -> int:
    for index, detail in enumerate(details):
        shape = detail.get("shape")
        if shape is not None and int(np.prod(shape)) == 521:
            return index
    return 0


def _infer(interpreter, input_detail, output_detail, window: np.ndarray, scale: float) -> np.ndarray:
    wave = window * scale
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        data = wave.astype(np.float32)
    elif dtype == np.int8:
        scale, zero = input_detail["quantization"]
        scale = scale or 1.0
        data = np.clip(np.round(wave / scale + zero), -128, 127).astype(np.int8)
    else:
        data = wave.astype(dtype)
    interpreter.set_tensor(input_detail["index"], data.reshape(input_detail["shape"]))
    interpreter.invoke()
    scores = interpreter.get_tensor(output_detail["index"]).astype(np.float32).reshape(-1)
    if scores.size != 521:
        raise RuntimeError("YAMNet вернул неожиданный размер ответа")
    return scores


def _classify(scores: np.ndarray) -> tuple[str, float] | None:
    assert _LABELS is not None and _MUTE_INDEX is not None
    if not _MUTE_INDEX:
        return None
    best_i = max(_MUTE_INDEX, key=lambda i: float(scores[i]))
    event_score = float(scores[best_i])
    speech_score = float(scores[_SPEECH_INDEX])
    strong = event_score >= 0.55
    clear = event_score >= 0.3 and event_score >= speech_score
    if not (strong or clear):
        return None
    return _LABELS[best_i], event_score


def _merge(hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    hits = sorted(hits, key=lambda item: item["start"])
    merged = [dict(hits[0])]
    for item in hits[1:]:
        last = merged[-1]
        if item["start"] <= last["end"]:
            last["end"] = max(last["end"], item["end"])
            if item["score"] > last["score"]:
                last["score"] = item["score"]
                last["label"] = item["label"]
        else:
            merged.append(dict(item))
    return merged


def _download(url: str, dest: Path) -> None:
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
