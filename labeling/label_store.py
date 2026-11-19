"""Persistent recording labels kept in a CSV file."""
import csv
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

LABELS = ("cat", "not_cat", "unsure")
FIELDS = ("file", "label", "labeled_at")


class LabelStore:
    """Maps recording file name -> label; every change is written to disk atomically."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._rows: Dict[str, dict] = {}
        if path.exists():
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("label") in LABELS:
                        self._rows[row["file"]] = row

    def get(self, file_name: str) -> Optional[str]:
        row = self._rows.get(file_name)
        return row["label"] if row else None

    def all(self) -> Dict[str, str]:
        return {name: row["label"] for name, row in self._rows.items()}

    def set(self, file_name: str, label: Optional[str]) -> None:
        """Set a label, or remove it when ``label`` is None."""
        if label is not None and label not in LABELS:
            raise ValueError(f"Unknown label {label!r}, expected one of {LABELS}")
        with self._lock:
            if label is None:
                self._rows.pop(file_name, None)
            else:
                self._rows[file_name] = {
                    "file": file_name,
                    "label": label,
                    "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                for name in sorted(self._rows):
                    writer.writerow(self._rows[name])
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise
