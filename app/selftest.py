"""Проверка щелчков, вздохов и заглушения без нейросетей."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from app.stages.breaths import find_breaths
from app.stages.declick import declick_block
from app.stages.gain import apply_gains


def test_click_removed_and_length_kept() -> None:
    sr = 48_000
    t = np.arange(sr, dtype=np.float32) / sr
    wave = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    wave[sr // 2] = 0.8
    fixed, found = declick_block(wave[:, None], sr)
    assert len(fixed) == len(wave)
    assert found, "щелчок не найден"
    assert abs(float(fixed[sr // 2, 0])) < 0.2


def test_plosive_kept() -> None:
    sr = 48_000
    wave = np.zeros(sr, dtype=np.float32)
    burst = np.sin(2 * np.pi * 1500 * np.arange(int(0.02 * sr)) / sr).astype(np.float32)
    fade = int(0.005 * sr)
    burst[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    burst[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    wave[sr // 3 : sr // 3 + len(burst)] = 0.4 * burst
    fixed, found = declick_block(wave[:, None], sr)
    assert not found
    assert np.allclose(fixed[:, 0], wave)


def test_breath_ducked_not_cut() -> None:
    sr = 48_000
    speech = (0.2 * np.sin(2 * np.pi * 150 * np.arange(sr) / sr)).astype(np.float32)
    breath = (0.08 * np.sin(2 * np.pi * 400 * np.arange(int(0.4 * sr)) / sr)).astype(np.float32)
    silence = np.zeros(int(0.2 * sr), dtype=np.float32)
    audio = np.concatenate([speech, silence, breath, silence, speech])
    path = __import__("pathlib").Path("/tmp/auphonic-breath.wav")
    sf.write(path, audio, sr, subtype="FLOAT")
    regions = [(0, len(speech)), (len(audio) - len(speech), len(audio))]
    breaths, speech_rms = find_breaths(path, sr, regions)
    assert speech_rms > 0.05
    assert breaths, "вздох не найден"
    before = breaths[0]["rms_before"]
    apply_gains(path, sr, breaths)
    after = breaths[0]
    data, _ = sf.read(path, dtype="float32")
    assert len(data) == len(audio)
    span = after["end"] - after["start"]
    piece = data[after["start"] + span // 5 : after["end"] - span // 5]
    level = float(np.sqrt(np.mean(piece**2)))
    assert level < max(1e-4, before * 0.02)
    assert np.max(np.abs(data[: len(speech)] - speech)) < 1e-5
    assert np.max(np.abs(data[-len(speech) :] - speech)) < 1e-5


def test_glued_inhale_removed_and_vowel_kept() -> None:
    sr = 48_000
    inhale = (0.04 * np.random.default_rng(0).standard_normal(int(0.35 * sr))).astype(np.float32)
    tone = (0.2 * np.sin(2 * np.pi * 150 * np.arange(sr) / sr)).astype(np.float32)
    audio = np.concatenate([inhale, tone])
    path = __import__("pathlib").Path("/tmp/auphonic-glued.wav")
    sf.write(path, audio, sr, subtype="FLOAT")
    breaths, _ = find_breaths(path, sr, [(0, len(audio))])
    assert breaths, "приклеенный вдох не найден"
    assert breaths[0]["end"] <= len(inhale) + int(0.05 * sr)
    apply_gains(path, sr, breaths)
    data, _ = sf.read(path, dtype="float32")
    assert len(data) == len(audio)
    muted = data[int(0.05 * sr) : len(inhale) // 2]
    assert float(np.sqrt(np.mean(muted**2))) < 1e-3
    kept = data[len(inhale) + int(0.2 * sr) : len(inhale) + int(0.6 * sr)]
    assert np.allclose(kept, tone[int(0.2 * sr) : int(0.6 * sr)], atol=1e-5)


def test_internal_inhale_between_words() -> None:
    sr = 48_000
    tone = (0.2 * np.sin(2 * np.pi * 150 * np.arange(sr) / sr)).astype(np.float32)
    inhale = (0.012 * np.random.default_rng(1).standard_normal(int(0.35 * sr))).astype(np.float32)
    audio = np.concatenate([tone, inhale, tone])
    path = __import__("pathlib").Path("/tmp/auphonic-valley.wav")
    sf.write(path, audio, sr, subtype="FLOAT")
    breaths, _ = find_breaths(path, sr, [(0, len(audio))])
    assert any(item["start"] >= len(tone) - int(0.05 * sr) and item["end"] <= 2 * len(tone) + int(0.05 * sr) for item in breaths)
    apply_gains(path, sr, breaths)
    data, _ = sf.read(path, dtype="float32")
    assert len(data) == len(audio)
    mid = data[len(tone) + int(0.18 * sr) : len(tone) + len(inhale) - int(0.08 * sr)]
    assert float(np.sqrt(np.mean(mid**2))) < 1e-3
    assert np.allclose(data[int(0.2 * sr) : int(0.6 * sr)], tone[int(0.2 * sr) : int(0.6 * sr)], atol=1e-4)


def test_voiced_ending_kept() -> None:
    sr = 48_000
    tone = (0.2 * np.sin(2 * np.pi * 150 * np.arange(int(0.5 * sr)) / sr)).astype(np.float32)
    decay = (0.05 * np.sin(2 * np.pi * 150 * np.arange(int(0.16 * sr)) / sr)).astype(np.float32)
    breath = (0.012 * np.random.default_rng(2).standard_normal(int(0.40 * sr))).astype(np.float32)
    audio = np.concatenate([tone, decay, breath])
    path = __import__("pathlib").Path("/tmp/auphonic-ending.wav")
    sf.write(path, audio, sr, subtype="FLOAT")
    breaths, _ = find_breaths(path, sr, [(0, len(audio))])
    apply_gains(path, sr, breaths)
    data, _ = sf.read(path, dtype="float32")
    assert len(data) == len(audio)
    ending = data[len(tone) : len(tone) + len(decay)]
    assert np.allclose(ending, decay, atol=1e-3)
    muted = data[len(tone) + len(decay) + int(0.16 * sr) : len(audio) - int(0.05 * sr)]
    assert float(np.sqrt(np.mean(muted**2))) < 1e-3


def main() -> None:
    test_click_removed_and_length_kept()
    test_plosive_kept()
    test_breath_ducked_not_cut()
    test_glued_inhale_removed_and_vowel_kept()
    test_internal_inhale_between_words()
    test_voiced_ending_kept()
    print("selftest ok")


if __name__ == "__main__":
    main()
