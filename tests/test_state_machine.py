"""Decision logic of the running state, with hardware and sound replaced by fakes."""
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def sm(monkeypatch):
    played = []
    fake_au = types.ModuleType("utils.audio_utils")
    fake_au.play_a_sound = lambda: None
    fake_au.play_d_sound = lambda: None
    fake_au.play_cat_sound = lambda: played.append("cat")
    fake_au.check_threshold = lambda chunk, norm_tsh: chunk.max() >= norm_tsh * 32767
    monkeypatch.setitem(sys.modules, "utils.audio_utils", fake_au)
    monkeypatch.delitem(sys.modules, "utils.state_machine", raising=False)
    import utils.state_machine as module
    module.played = played
    return module


class FakeRecorder:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stop_stream(self):
        pass

    def read_chank(self, chunk_size=1024):
        return [None, next(self.chunks, np.zeros(chunk_size, dtype=np.int16))]


class FakeDetector:
    def __init__(self, verdict):
        self.verdict = verdict

    def run_detection(self, chunk):
        return np.array([self.verdict])


class FakeSaver:
    def __init__(self):
        self.saved = []

    def save(self, record, file_name=None):
        self.saved.append((file_name, record.size))
        return "/tmp/x.wav"


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def send_file(self, path, metadata=None):
        self.sent.append(metadata)
        return self.response


def run_once(sm, svm_verdict, server_response):
    loud = np.full(1024, 32000, dtype=np.int16)
    ctx = {"audio_rec": FakeRecorder([loud]), "detector": FakeDetector(svm_verdict),
           "rec_saver": FakeSaver(), "cat_detector_client": FakeClient(server_response)}
    state = sm.StateRunning(ctx)
    state.run()
    return ctx


def test_server_verdict_alone_triggers_alarm_even_if_svm_rejects(sm):
    ctx = run_once(sm, 0, {"cat_meowing_detected": True, "cat_score": 0.9})
    assert ctx["cat_detector_client"].sent == [{"svm": "0"}]
    assert sm.played == ["cat"]


def test_no_alarm_when_server_says_no_cat_even_if_svm_says_cat(sm):
    ctx = run_once(sm, 1, {"cat_meowing_detected": False, "cat_score": 0.01})
    assert sm.played == []
    assert [name for name, _ in ctx["rec_saver"].saved] == ["tmp.wav"]


def test_alarm_when_svm_and_server_agree(sm):
    ctx = run_once(sm, 1, {"cat_meowing_detected": True, "cat_score": 0.9})
    assert ctx["cat_detector_client"].sent == [{"svm": "1"}]
    assert sm.played == ["cat"]
    # tmp clip for the server + permanent copy; ~2 s of audio (1 s pre + 1 s post roll)
    assert [name for name, _ in ctx["rec_saver"].saved] == ["tmp.wav", None]
    assert ctx["rec_saver"].saved[0][1] == 1024 + 16 * 1024


def test_no_alarm_when_server_unavailable(sm):
    run_once(sm, 1, None)
    assert sm.played == []
