"""HTTP API used by the Raspberry Pi to confirm cat meows.

Run: ``.venv/bin/uvicorn cat_detection_server:app --host 0.0.0.0 --port 8981``
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional, Protocol, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from audio_io import InvalidAudioError, decode_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cat_detection_server")


class Detector(Protocol):
    def detect(self, waveform: np.ndarray, sample_rate: int) -> Tuple[bool, float]: ...


def create_app(detector: Optional[Detector] = None) -> FastAPI:
    """Build the API. Without ``detector`` the EfficientAT model is loaded at startup."""
    state = {"detector": detector}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["detector"] is None:
            from cat_model import CatDetector  # heavy import (torch), only when serving

            state["detector"] = CatDetector()
            logger.info("Model loaded")
        yield

    app = FastAPI(title="CatCom cat detection server", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": state["detector"] is not None}

    # Sync handler: FastAPI runs it in a thread pool, so CPU-bound inference
    # does not block the event loop.
    @app.post("/upload/")
    def upload_file(file: UploadFile = File(...)) -> dict:
        try:
            waveform, sample_rate = decode_wav(file.file.read())
        except InvalidAudioError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        is_cat_meowing, score = state["detector"].detect(waveform, sample_rate)
        logger.info("%s: %.2f s, cat score %.3f -> %s", file.filename,
                    waveform.size / sample_rate, score,
                    "CAT" if is_cat_meowing else "no cat")
        return {"cat_meowing_detected": bool(is_cat_meowing), "cat_score": float(score)}

    return app


app = create_app()
