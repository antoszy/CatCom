"""Cat meow detector built on EfficientAT ``mn10_as`` (AudioSet pretrained)."""
import hashlib
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio

from cat_scoring import cat_score, resolve_class_indices
from efficientat.labels import load_labels
from efficientat.mn_model import get_model
from efficientat.preprocess import AugmentMelSTFT

logger = logging.getLogger(__name__)

MODEL_NAME = "mn10_as"
WEIGHTS_URL = ("https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
               "mn10_as_mAP_471.pt")
WEIGHTS_SHA256 = "0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e"
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("CATCOM_WEIGHTS_DIR",
                                          Path.home() / ".cache" / "catcom"))

# Preprocessing must match how the pretrained model was trained
# (upstream inference.py defaults).
MODEL_SAMPLE_RATE = 32000
N_MELS = 128
WIN_LENGTH = 800
HOP_SIZE = 320
# The model was trained on 10 s AudioSet clips. On much shorter input it
# produces saturated nonsense (e.g. 1.0 for dozens of classes on a 2 s clip),
# so short clips are zero-padded to this length, as upstream
# windowed_inference.py does for its last window.
MIN_CLIP_S = 10.0

# Chosen on 90 hand-labelled 0.45 s clips (old pipeline's positives, 56 cat /
# 34 not cat): precision 0.98, recall 0.71. Favours few false alarms over
# catching every meow. Re-check once 2 s clips have been labelled.
DEFAULT_THRESHOLD = float(os.environ.get("CATCOM_THRESHOLD", "0.5"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> Path:
    """Return the path to verified mn10_as weights, downloading them if needed."""
    path = weights_dir / Path(WEIGHTS_URL).name
    if not path.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".part")
        logger.info("Downloading %s to %s", WEIGHTS_URL, path)
        urllib.request.urlretrieve(WEIGHTS_URL, tmp_path)
        tmp_path.replace(path)
    actual = _sha256(path)
    if actual != WEIGHTS_SHA256:
        raise RuntimeError(f"Checksum mismatch for {path}: {actual} != {WEIGHTS_SHA256}. "
                           "Delete the file to re-download it.")
    return path


def pad_to_min_length(wav: torch.Tensor, min_samples: int) -> torch.Tensor:
    """Zero-pad a 1-D waveform at the end up to ``min_samples``; longer input is unchanged."""
    missing = min_samples - wav.shape[-1]
    if missing <= 0:
        return wav
    return torch.nn.functional.pad(wav, (0, missing))


class CatDetector:
    """Scores audio clips for cat vocalisations (Cat / Meow / Caterwaul)."""

    def __init__(self, weights_path: Optional[Path] = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 device: str = "cpu"):
        self.threshold = threshold
        self.device = torch.device(device)
        self.labels = load_labels()
        self.cat_class_indices = resolve_class_indices(self.labels)

        self.model = get_model(width_mult=1.0, pretrained_name=None)
        state_dict = torch.load(weights_path or ensure_weights(),
                                map_location="cpu", weights_only=True)
        # strict: a silent partial load would give plausible-looking garbage.
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

        self.mel = AugmentMelSTFT(n_mels=N_MELS, sr=MODEL_SAMPLE_RATE,
                                  win_length=WIN_LENGTH, hopsize=HOP_SIZE)
        self.mel.to(self.device).eval()
        # Requests are handled in a thread pool; keep inference serialized.
        self._lock = threading.Lock()

    def predict(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return per-class probabilities (527 AudioSet classes) for a mono clip."""
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Expected a non-empty mono waveform, got shape {waveform.shape}")
        wav = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        if sample_rate != MODEL_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sample_rate, MODEL_SAMPLE_RATE)
        wav = pad_to_min_length(wav, int(MIN_CLIP_S * MODEL_SAMPLE_RATE))
        with self._lock, torch.no_grad():
            spec = self.mel(wav.unsqueeze(0).to(self.device))
            logits, _ = self.model(spec.unsqueeze(0))
        return torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()

    def detect(self, waveform: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """Return (is_cat_meowing, score) for a mono clip."""
        probs = self.predict(waveform, sample_rate)
        score = cat_score(probs.reshape(1, -1), self.cat_class_indices)
        return score > self.threshold, score
