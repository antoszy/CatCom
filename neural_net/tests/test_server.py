import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from cat_detection_server import create_app


class FakeDetector:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def detect(self, waveform, sample_rate):
        self.calls.append((waveform.size, sample_rate))
        return self.result


def wav_bytes(n_samples: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(n_samples, dtype=np.int16).tobytes())
    return buf.getvalue()


def test_upload_returns_detection_and_score():
    detector = FakeDetector((True, 0.8))
    with TestClient(create_app(detector)) as client:
        response = client.post("/upload/", files={"file": ("tmp.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    assert response.json() == {"cat_meowing_detected": True, "cat_score": 0.8}
    assert detector.calls == [(16000, 16000)]


def test_invalid_audio_is_400_and_skips_model():
    detector = FakeDetector((True, 1.0))
    with TestClient(create_app(detector)) as client:
        response = client.post("/upload/", files={"file": ("x.wav", b"junk", "audio/wav")})
    assert response.status_code == 400
    assert detector.calls == []


def test_health():
    with TestClient(create_app(FakeDetector((False, 0.0)))) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
