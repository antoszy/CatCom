"""Sends recorded clips to the detection server in a background thread.

The main loop must keep reading the microphone: while it waited for the
server (~1-2 s per clip) audio was dropped and meows were missed. Here a
single worker thread saves each clip, sends it, and rings the alarm when the
server says "cat".

The alarm is a cat sound, so the microphone would hear it and trigger itself.
While it plays and for ``cooldown_s`` afterwards the sender reports
``is_muted()`` (the main loop ignores loud audio then), and clips captured
before the alarm ended never ring it again.
"""
import queue
import threading
import time
from typing import Callable, NamedTuple, Optional

import numpy as np

REC_FILE_NAME = "tmp.wav"
# Room echo and audio buffers still carry the alarm shortly after it ends.
DEFAULT_COOLDOWN_S = 3.0
# Clips arrive every 1-2 s in a noisy room; if the server falls behind, the
# oldest clips are dropped so the newest (most relevant) audio is checked.
DEFAULT_MAX_PENDING = 3


class Clip(NamedTuple):
    record: np.ndarray
    svm_says_cat: bool
    captured_at: float  # clock() when the loud chunk was heard


class ClipSender:
    """Background save -> send -> alarm pipeline for recorded clips."""

    def __init__(self, client, rec_saver, play_alarm: Callable[[], None],
                 cooldown_s: float = DEFAULT_COOLDOWN_S,
                 max_pending: int = DEFAULT_MAX_PENDING,
                 clock: Callable[[], float] = time.monotonic):
        self._client = client
        self._rec_saver = rec_saver
        self._play_alarm = play_alarm
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._queue = queue.Queue(maxsize=max_pending)
        self._lock = threading.Lock()
        self._playing = False
        self._alarm_ended_at = float("-inf")
        self._active = True
        self._worker = threading.Thread(target=self._work, name="clip-sender", daemon=True)
        self._worker.start()

    def now(self) -> float:
        return self._clock()

    def submit(self, record: np.ndarray, svm_says_cat: bool, captured_at: float) -> None:
        """Queue a clip; never blocks (drops the oldest pending clip when full)."""
        clip = Clip(record, svm_says_cat, captured_at)
        while True:
            try:
                self._queue.put_nowait(clip)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    print("Clip sender: server falling behind, dropped oldest clip")
                except queue.Empty:
                    pass

    def is_muted(self) -> bool:
        """True while the alarm plays and for the cooldown after it."""
        with self._lock:
            return self._playing or self.now() < self._alarm_ended_at + self._cooldown_s

    def set_active(self, active: bool) -> None:
        """Inactive: pending clips are dropped and no alarm rings (standby)."""
        with self._lock:
            self._active = active
        if not active:
            self._drain()

    def wait_until_idle(self) -> None:
        """Block until every submitted clip is processed (used in tests)."""
        self._queue.join()

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    def _work(self) -> None:
        while True:
            clip = self._queue.get()
            try:
                self._process(clip)
            except Exception as exc:  # the worker must survive any single clip
                print(f"Clip sender: failed to process clip: {exc!r}")
            finally:
                self._queue.task_done()

    def _process(self, clip: Clip) -> None:
        rec_path = self._rec_saver.save(clip.record, REC_FILE_NAME)
        response = self._client.send_file(
            rec_path, metadata={"svm": str(int(clip.svm_says_cat))})
        print(f"Detection server response: {response}")
        # The server alone decides about the alarm; the SVM verdict is only
        # sent along for analysis. On 2026-11-20 the SVM rejected all 7 meows
        # the server caught and flagged 50 non-cat sounds, so it can't gate.
        # No server answer -> no alarm.
        if not (response and response.get("cat_meowing_detected")):
            return
        if not self._start_alarm(clip):
            return
        print("Cat detected")
        try:
            self._play_alarm()
        finally:
            with self._lock:
                self._playing = False
                self._alarm_ended_at = self.now()
        # Permanent copy for verification by hand.
        self._rec_saver.save(clip.record)

    def _start_alarm(self, clip: Clip) -> bool:
        with self._lock:
            if not self._active:
                return False
            if clip.captured_at <= self._alarm_ended_at + self._cooldown_s:
                print("Cat detected, but clip predates the last alarm: not ringing again")
                return False
            self._playing = True
            return True
