"""Keeps every clip sent by the Pi, with its scores, for later labelling."""
import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FIELDS = ("file", "received_at", "cat_score", "detected", "svm")


class CandidateStore:
    """Writes clips as ``<timestamp>.wav`` plus one CSV row per clip."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.index_path = directory / "candidates.csv"
        self._lock = threading.Lock()

    def save(self, wav_bytes: bytes, cat_score: float, detected: bool,
             svm: Optional[str] = None) -> Path:
        now = datetime.now()
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._unique_path(now)
            path.write_bytes(wav_bytes)
            write_header = not self.index_path.exists()
            with open(self.index_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "file": path.name,
                    "received_at": now.isoformat(timespec="milliseconds"),
                    "cat_score": f"{cat_score:.4f}",
                    "detected": int(detected),
                    "svm": "" if svm is None else svm,
                })
        return path

    def _unique_path(self, now: datetime) -> Path:
        base = now.strftime("cand_%Y%m%d_%H%M%S_%f")
        path = self.directory / f"{base}.wav"
        suffix = 1
        while path.exists():
            path = self.directory / f"{base}_{suffix}.wav"
            suffix += 1
        return path
