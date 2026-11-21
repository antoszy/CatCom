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


class FakeSender:
    def __init__(self, muted=False):
        self.muted = muted
        self.submitted = []
        self.active = []

    def now(self):
        return 42.0

    def is_muted(self):
        return self.muted

    def set_active(self, active):
        self.active.append(active)

    def submit(self, record, svm_says_cat, captured_at):
        self.submitted.append((record.size, svm_says_cat, captured_at))


LOUD = np.full(1024, 32000, dtype=np.int16)
QUIET = np.zeros(1024, dtype=np.int16)


def make_state(sm, chunks, svm_verdict=0, sender=None):
    ctx = {"audio_rec": FakeRecorder(chunks), "detector": FakeDetector(svm_verdict),
           "clip_sender": sender or FakeSender()}
    return sm.StateRunning(ctx), ctx["clip_sender"]


@pytest.mark.parametrize("svm_verdict", [0, 1])
def test_loud_chunk_submits_2s_clip_with_svm_verdict(sm, svm_verdict):
    state, sender = make_state(sm, [LOUD], svm_verdict)
    state.run()
    # 1 s pre-roll (here only the trigger chunk) + 1 s post-roll
    assert sender.submitted == [(1024 + 16 * 1024, bool(svm_verdict), 42.0)]
    assert sender.active == [True]


def test_quiet_chunk_submits_nothing(sm):
    state, sender = make_state(sm, [QUIET])
    state.run()
    assert sender.submitted == []


def test_muted_sender_ignores_loud_audio_and_clears_pre_roll(sm):
    sender = FakeSender(muted=True)
    state, _ = make_state(sm, [QUIET, LOUD], sender=sender)
    state.run()
    state.run()
    assert sender.submitted == [] and len(state.pre_roll) == 0


def test_standby_deactivates_sender(sm):
    sender = FakeSender()
    sm.StateStandby({"clip_sender": sender})
    assert sender.active == [False]
