"""Tests with the real EfficientAT model (needs downloaded weights, ~20 MB)."""
from pathlib import Path

import numpy as np
import pytest
import torch

import cat_model
from audio_io import decode_wav

SOUNDS = Path(__file__).resolve().parents[2] / "sounds"
WEIGHTS = cat_model.DEFAULT_WEIGHTS_DIR / Path(cat_model.WEIGHTS_URL).name


def test_pad_to_min_length():
    assert cat_model.pad_to_min_length(torch.ones(3), 5).tolist() == [1, 1, 1, 0, 0]
    assert cat_model.pad_to_min_length(torch.ones(6), 5).shape[0] == 6


@pytest.fixture(scope="module")
def detector():
    if not WEIGHTS.exists():
        pytest.skip(f"weights not downloaded ({WEIGHTS}); run setup_server.sh")
    return cat_model.CatDetector()


def test_activation_beep_is_not_a_cat(detector):
    waveform, sr = decode_wav((SOUNDS / "a.wav").read_bytes())
    is_cat, score = detector.detect(waveform, sr)
    assert not is_cat and score < 0.1


def test_short_clip_does_not_saturate(detector):
    # Regression: unpadded short clips gave probability ~1.0 for dozens of classes.
    waveform, sr = decode_wav((SOUNDS / "d.wav").read_bytes())
    probs = detector.predict(waveform[: sr // 2], sr)
    assert probs.shape == (527,)
    assert (probs > 0.9).sum() == 0
