"""Web page for labelling recordings as cat / not cat.

Run (from repo root, uses the server venv for the optional model scores):
    neural_net/.venv/bin/python labeling/label_app.py --recordings data/candidates
Then open http://127.0.0.1:8982. Labels go to <recordings>/labels.csv.
New recordings appear on page reload, no restart needed.
"""
import argparse
import csv
import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from label_store import LABELS, LabelStore

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger("label_app")


class LabelRequest(BaseModel):
    label: Optional[str] = None  # None removes the label


Scorer = Callable[[Path], Optional[float]]


class ModelScorer:
    """Cat score from the EfficientAT detector; the model is loaded on first use."""

    def __init__(self):
        self._detector = None
        self._lock = threading.Lock()

    def __call__(self, path: Path) -> Optional[float]:
        sys.path.insert(0, str(REPO_ROOT / "neural_net"))
        from audio_io import InvalidAudioError, decode_wav

        with self._lock:
            if self._detector is None:
                from cat_model import CatDetector
                self._detector = CatDetector()
            try:
                waveform, sr = decode_wav(path.read_bytes())
            except InvalidAudioError as exc:
                logger.warning("No score for %s: %s", path.name, exc)
                return None
            return round(self._detector.detect(waveform, sr)[1], 3)


def read_index_scores(recordings_dir: Path) -> Dict[str, float]:
    """Scores the detection server stored with the clips (candidates.csv), if any."""
    index = recordings_dir / "candidates.csv"
    if not index.exists():
        return {}
    with open(index, newline="") as f:
        return {row["file"]: round(float(row["cat_score"]), 3)
                for row in csv.DictReader(f) if row.get("cat_score")}


def create_app(recordings_dir: Path, labels_path: Path,
               scorer: Optional[Scorer] = None) -> FastAPI:
    store = LabelStore(labels_path)
    score_cache: Dict[str, Optional[float]] = {}
    app = FastAPI(title="CatCom labelling")

    def list_files():
        return sorted(recordings_dir.glob("*.wav"))

    def score_of(path: Path, index_scores: Dict[str, float]) -> Optional[float]:
        if path.name in index_scores:
            return index_scores[path.name]
        if scorer is None:
            return None
        if path.name not in score_cache:
            score_cache[path.name] = scorer(path)
        return score_cache[path.name]

    def require_known(name: str) -> Path:
        # Only names of existing recordings are accepted: no path traversal.
        for path in list_files():
            if path.name == name:
                return path
        raise HTTPException(status_code=404, detail=f"Unknown recording {name}")

    @app.get("/api/recordings")
    def recordings() -> dict:
        index_scores = read_index_scores(recordings_dir)
        return {
            "labels": list(LABELS),
            "labels_file": str(labels_path),
            "items": [{"file": p.name, "score": score_of(p, index_scores),
                       "label": store.get(p.name)} for p in list_files()],
        }

    @app.get("/audio/{name}")
    def audio(name: str) -> FileResponse:
        return FileResponse(require_known(name), media_type="audio/wav")

    @app.put("/api/labels/{name}")
    def set_label(name: str, body: LabelRequest) -> dict:
        require_known(name)
        try:
            store.set(name, body.label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"file": name, "label": body.label}

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, default=REPO_ROOT / "data" / "candidates")
    parser.add_argument("--labels", type=Path, default=None,
                        help="CSV with labels (default: <recordings>/labels.csv)")
    parser.add_argument("--no-scores", action="store_true", help="skip the model scores")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8982)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not args.recordings.is_dir():
        parser.error(f"Recordings directory not found: {args.recordings}")
    labels_path = args.labels or args.recordings / "labels.csv"
    logger.info("Recordings in %s, labels in %s", args.recordings, labels_path)
    scorer = None if args.no_scores else ModelScorer()
    uvicorn.run(create_app(args.recordings, labels_path, scorer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
