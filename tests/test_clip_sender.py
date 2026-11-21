"""Background sending and alarm logic of ClipSender, with fake server/saver/clock."""
import threading

import numpy as np
import pytest

from utils.clip_sender import ClipSender

CAT = {"cat_meowing_detected": True, "cat_score": 0.9}
NO_CAT = {"cat_meowing_detected": False, "cat_score": 0.01}


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def send_file(self, path, metadata=None):
        self.sent.append(metadata)
        return self.responses.pop(0) if self.responses else None


class FakeSaver:
    def __init__(self):
        self.saved = []

    def save(self, record, file_name=None):
        self.saved.append(file_name)
        return "/tmp/x.wav"


def make(responses, clock=None, **kwargs):
    clock = clock or FakeClock()
    played = []
    client, saver = FakeClient(responses), FakeSaver()
    sender = ClipSender(client, saver, play_alarm=lambda: played.append(clock()),
                        clock=clock, **kwargs)
    return sender, client, saver, played, clock


RECORD = np.zeros(10, dtype=np.int16)


def test_server_cat_rings_alarm_and_keeps_copy_even_if_svm_rejects():
    sender, client, saver, played, clock = make([CAT])
    sender.submit(RECORD, False, clock())
    sender.wait_until_idle()
    assert client.sent == [{"svm": "0"}]
    assert len(played) == 1
    assert saver.saved == ["tmp.wav", None]


@pytest.mark.parametrize("response", [NO_CAT, None])
def test_no_alarm_without_server_cat_verdict(response):
    sender, client, saver, played, clock = make([response])
    sender.submit(RECORD, True, clock())
    sender.wait_until_idle()
    assert client.sent == [{"svm": "1"}]
    assert played == [] and saver.saved == ["tmp.wav"]


def test_muted_during_cooldown_after_alarm():
    sender, _, _, played, clock = make([CAT], cooldown_s=3.0)
    assert not sender.is_muted()
    sender.submit(RECORD, False, clock())
    sender.wait_until_idle()
    clock.t += 2.9
    assert sender.is_muted()
    clock.t += 0.2
    assert not sender.is_muted()


def test_muted_while_alarm_plays():
    started, release = threading.Event(), threading.Event()

    def play():
        started.set()
        release.wait(5)

    clock = FakeClock()
    sender = ClipSender(FakeClient([CAT]), FakeSaver(), play_alarm=play, clock=clock)
    sender.submit(RECORD, False, clock())
    assert started.wait(5)
    clock.t += 1000  # long past any cooldown, still playing
    assert sender.is_muted()
    release.set()
    sender.wait_until_idle()
    assert sender.is_muted()  # cooldown starts when the sound ends


def test_clip_captured_before_alarm_does_not_ring_again():
    sender, _, _, played, clock = make([CAT, CAT], cooldown_s=3.0)
    first, second = clock(), clock() + 0.5
    clock.t += 1.0
    sender.submit(RECORD, False, first)
    sender.submit(RECORD, False, second)
    sender.wait_until_idle()
    assert len(played) == 1


def test_clip_after_cooldown_rings_again():
    sender, _, _, played, clock = make([CAT, CAT], cooldown_s=3.0)
    sender.submit(RECORD, False, clock())
    sender.wait_until_idle()
    clock.t += 3.5
    sender.submit(RECORD, False, clock())
    sender.wait_until_idle()
    assert len(played) == 2


def test_inactive_sender_does_not_ring():
    sender, client, _, played, clock = make([CAT])
    sender.set_active(False)
    sender.submit(RECORD, False, clock())
    sender.wait_until_idle()
    assert played == []


def test_full_queue_drops_oldest_clip_without_blocking():
    entered, release = threading.Event(), threading.Event()

    class SlowClient(FakeClient):
        def send_file(self, path, metadata=None):
            entered.set()
            release.wait(5)
            return super().send_file(path, metadata)

    client = SlowClient([])
    sender = ClipSender(client, FakeSaver(), play_alarm=lambda: None, max_pending=2)
    sender.submit(RECORD, True, 0.0)
    assert entered.wait(5)  # first clip is in flight
    for svm in (False, False, True, True):  # 4 clips for 2 free slots
        sender.submit(RECORD, svm, 0.0)
    release.set()
    sender.wait_until_idle()
    assert len(client.sent) == 3
    assert client.sent[-2:] == [{"svm": "1"}, {"svm": "1"}]


def test_worker_survives_failing_clip():
    class BrokenSaver(FakeSaver):
        def save(self, record, file_name=None):
            if not self.saved:
                self.saved.append("boom")
                raise OSError("disk full")
            return super().save(record, file_name)

    client = FakeClient([NO_CAT])
    sender = ClipSender(client, BrokenSaver(), play_alarm=lambda: None)
    sender.submit(RECORD, False, 0.0)
    sender.submit(RECORD, False, 0.0)
    sender.wait_until_idle()
    assert client.sent == [{"svm": "0"}]
