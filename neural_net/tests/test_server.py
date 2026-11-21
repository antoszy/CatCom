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


def test_upload_stores_candidate_with_svm_verdict(tmp_path):
    from candidate_store import CandidateStore

    store = CandidateStore(tmp_path)
    app = create_app(FakeDetector((False, 0.2)), candidates=store)
    with TestClient(app) as client:
        response = client.post("/upload/", files={"file": ("tmp.wav", wav_bytes(), "audio/wav")},
                               data={"svm": "1"})
    assert response.status_code == 200
    saved = list(tmp_path.glob("cand_*.wav"))
    assert len(saved) == 1 and saved[0].read_bytes() == wav_bytes()
    assert "0.2000,0,1" in store.index_path.read_text()


def test_storage_failure_does_not_break_detection(tmp_path):
    class BrokenStore:
        def save(self, *args):
            raise OSError("disk full")

    with TestClient(create_app(FakeDetector((True, 0.9)), candidates=BrokenStore())) as client:
        response = client.post("/upload/", files={"file": ("tmp.wav", wav_bytes(), "audio/wav")},
                               data={"svm": "1"})
    assert response.status_code == 200 and response.json()["cat_meowing_detected"] is True


def test_low_score_clip_rejected_by_svm_is_not_stored(tmp_path):
    from candidate_store import CandidateStore

    store = CandidateStore(tmp_path)
    for svm in ("0", None):
        data = {} if svm is None else {"svm": svm}
        with TestClient(create_app(FakeDetector((False, 0.049)), candidates=store)) as client:
            response = client.post("/upload/", files={"file": ("tmp.wav", wav_bytes(), "audio/wav")},
                                   data=data)
        assert response.status_code == 200
    assert list(tmp_path.glob("*")) == []


def test_svm_rejected_clip_with_high_score_is_stored(tmp_path):
    from candidate_store import CandidateStore

    store = CandidateStore(tmp_path)
    with TestClient(create_app(FakeDetector((False, 0.05)), candidates=store)) as client:
        client.post("/upload/", files={"file": ("tmp.wav", wav_bytes(), "audio/wav")},
                    data={"svm": "0"})
    assert len(list(tmp_path.glob("cand_*.wav"))) == 1
    assert "0.0500,0,0" in store.index_path.read_text()
