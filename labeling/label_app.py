"""Web page for labelling recordings as cat / not cat.

Run (from repo root, uses the server venv for the optional model scores):
    neural_net/.venv/bin/python labeling/label_app.py --recordings data/recordings
Then open http://127.0.0.1:8982
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


def compute_scores(files: List[Path]) -> Dict[str, float]:
    """Cat score of each recording from the EfficientAT detector."""
    sys.path.insert(0, str(REPO_ROOT / "neural_net"))
    from audio_io import InvalidAudioError, decode_wav
    from cat_model import CatDetector

    detector = CatDetector()
    scores = {}
    for path in files:
        try:
            waveform, sr = decode_wav(path.read_bytes())
        except InvalidAudioError as exc:
            logger.warning("Skipping score for %s: %s", path.name, exc)
            continue
        scores[path.name] = round(detector.detect(waveform, sr)[1], 3)
    return scores


def create_app(recordings_dir: Path, labels_path: Path,
               scores: Optional[Dict[str, float]] = None) -> FastAPI:
    files = sorted(recordings_dir.glob("*.wav"))
    names = {p.name for p in files}
    store = LabelStore(labels_path)
    scores = scores or {}
    app = FastAPI(title="CatCom labelling")

    def require_known(name: str) -> None:
        # Only files listed at startup are served: no path traversal via the name.
        if name not in names:
            raise HTTPException(status_code=404, detail=f"Unknown recording {name}")

    @app.get("/api/recordings")
    def recordings() -> dict:
        return {
            "labels": list(LABELS),
            "labels_file": str(labels_path),
            "items": [{"file": p.name, "score": scores.get(p.name), "label": store.get(p.name)}
                      for p in files],
        }

    @app.get("/audio/{name}")
    def audio(name: str) -> FileResponse:
        require_known(name)
        return FileResponse(recordings_dir / name, media_type="audio/wav")

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
    parser.add_argument("--recordings", type=Path, default=REPO_ROOT / "data" / "recordings")
    parser.add_argument("--labels", type=Path, default=None,
                        help="CSV with labels (default: <recordings>/../labels.csv)")
    parser.add_argument("--no-scores", action="store_true", help="skip the model scores")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8982)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not args.recordings.is_dir():
        parser.error(f"Recordings directory not found: {args.recordings}")
    labels_path = args.labels or args.recordings.parent / "labels.csv"
    files = sorted(args.recordings.glob("*.wav"))
    scores = {} if args.no_scores else compute_scores(files)
    logger.info("%d recordings, labels in %s", len(files), labels_path)
    uvicorn.run(create_app(args.recordings, labels_path, scores), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
