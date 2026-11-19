"""Keeps every clip sent by the Pi, with its scores, for later labelling."""
import csv
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

FIELDS = ("file", "received_at", "cat_score", "detected", "svm")
# Stop collecting before the clips can fill the disk the server shares with the OS.
DEFAULT_MAX_DISK_USED = 0.9


class CandidateStore:
    """Writes clips as ``<timestamp>.wav`` plus one CSV row per clip.

    ``disk_usage`` returns an object with ``total`` and ``used`` (as
    ``shutil.disk_usage``); injectable for tests.
    """

    def __init__(self, directory: Path, max_disk_used: float = DEFAULT_MAX_DISK_USED,
                 disk_usage: Callable[[Path], Any] = shutil.disk_usage):
        self.directory = directory
        self.index_path = directory / "candidates.csv"
        self.max_disk_used = max_disk_used
        self._disk_usage = disk_usage
        self._lock = threading.Lock()

    def save(self, wav_bytes: bytes, cat_score: float, detected: bool,
             svm: Optional[str] = None) -> Optional[Path]:
        """Store the clip; returns None (nothing written) when the disk is too full."""
        now = datetime.now()
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self._disk_too_full():
                return None
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

    def _disk_too_full(self) -> bool:
        usage = self._disk_usage(self.directory)
        used = usage.used / usage.total
        if used > self.max_disk_used:
            logger.warning("Disk %.0f%% full (limit %.0f%%), not storing candidate clip",
                           used * 100, self.max_disk_used * 100)
            return True
        return False

    def _unique_path(self, now: datetime) -> Path:
        base = now.strftime("cand_%Y%m%d_%H%M%S_%f")
        path = self.directory / f"{base}.wav"
        suffix = 1
        while path.exists():
            path = self.directory / f"{base}_{suffix}.wav"
            suffix += 1
        return path
