from contextlib import contextmanager, ExitStack
import time
from datetime import datetime
import numpy as np
import utils.audio_utils as au
from utils.ring_buffer import ChunkRingBuffer, chunks_for_duration

LOUD_TSH = 0.85
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
# Audio kept before the loud chunk, so the onset of a meow is not cut off.
# The loud (trigger) chunk itself is part of the pre-roll.
PRE_ROLL_S = 1.0
# Audio recorded after the loud chunk. Pre + post roll (~2 s) gives YAMNet
# (0.96 s window, 0.48 s hop) several full frames instead of one padded one.
POST_ROLL_S = 1.0


class State:
    def run(self):
        assert 0, "run not implemented"

    def change(self, ctx):
        assert 0, "change not implemented"


class StateStandby(State):
    def __init__(self, ctx):
        au.play_d_sound()
        print("Entering standby state")
        ctx["clip_sender"].set_active(False)
        print("Standby state")


    def run(self):
        time.sleep(0.2)

    def change(self, ctx):
        return StateRunning(ctx)


class StateRunning(State, ExitStack):
    def __init__(self, ctx):
        au.play_a_sound()
        print("Entering running state")
        super().__init__()
        super().__enter__()
        self.ctx = ctx
        ctx["clip_sender"].set_active(True)
        self.audio_rec = self.enter_context(ctx["audio_rec"])
        self.pre_roll = ChunkRingBuffer(
            chunks_for_duration(PRE_ROLL_S, CHUNK_SIZE, SAMPLE_RATE))
        self.post_roll_chunks = chunks_for_duration(POST_ROLL_S, CHUNK_SIZE, SAMPLE_RATE)
        print("Running state")

    def __del__(self):
        print("Exiting running state")
        self.audio_rec.stop_stream()

    def _record_with_pre_roll(self) -> np.ndarray:
        """Return pre-roll (incl. the trigger chunk) followed by freshly read post-roll."""
        post_roll = [self.audio_rec.read_chank(CHUNK_SIZE)[1]
                     for _ in range(self.post_roll_chunks)]
        record = np.concatenate([self.pre_roll.to_array()] + post_roll).astype(np.int16)
        # Next trigger must not reuse audio that has already been evaluated.
        self.pre_roll.clear()
        return record

    def run(self):
        audio_chunk = self.audio_rec.read_chank(CHUNK_SIZE)[1]
        sender = self.ctx["clip_sender"]
        if sender.is_muted():
            # Our own alarm is playing: keep it out of the detector and of
            # the pre-roll of the next clip.
            self.pre_roll.clear()
            return
        self.pre_roll.append(audio_chunk)
        if au.check_threshold(audio_chunk, norm_tsh=LOUD_TSH):
            captured_at = sender.now()
            cat_detect = self.ctx["detector"].run_detection(audio_chunk)
            print(f"Loud audio detected at: {datetime.now().time()} " +
                  f"Detector output = {cat_detect[0]} " +
                  f"Max audio = {max(audio_chunk) / np.iinfo(audio_chunk.dtype).max}")
            # Sending and the alarm run in the sender's thread, so the
            # microphone keeps being read while the server works.
            sender.submit(self._record_with_pre_roll(), bool(cat_detect[0]), captured_at)


    def change(self, ctx):
        return StateStandby(ctx)


class StateMachine:
    def __init__(self, audio_rec, detector, clip_sender, start_running=False):
        self.context = {
            "audio_rec": audio_rec,
            "detector": detector,
            "clip_sender": clip_sender,
        }
        initial_state = StateRunning if start_running else StateStandby
        self.currentState = initial_state(self.context)

    def change_state(self):
        self.currentState = self.currentState.change(self.context)

    def run(self):
        self.currentState.run()
