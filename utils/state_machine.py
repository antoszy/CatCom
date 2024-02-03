from contextlib import contextmanager, ExitStack
import time
from datetime import datetime
import numpy as np
import utils.audio_utils as au
import sys

LOUD_TSH = 0.85
REC_LENGTH = 6
REC_FILE_NAME = "tmp.wav"


class State:
    def run(self):
        assert 0, "run not implemented"

    def change(self, ctx):
        assert 0, "change not implemented"


class StateStandby(State):
    def __init__(self, ctx):
        au.play_d_sound()
        print("Entering standby state")
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
        self.audio_rec = self.enter_context(ctx["audio_rec"])
        print("Running state")

    def __del__(self):
        print("Exiting running state")
        self.audio_rec.stop_stream()

    def run(self):
        audio_chunk = self.audio_rec.read_chank()[1]
        if au.check_threshold(audio_chunk, norm_tsh=LOUD_TSH):
            cat_detect = self.ctx["detector"].run_detection(audio_chunk)
            print(f"Loud audio detected at: {datetime.now().time()} " +
                  f"Detector output = {cat_detect[0]} " +
                  f"Max audio = {max(audio_chunk) / np.iinfo(audio_chunk.dtype).max}")
            if cat_detect or True:
                # sys.stdout.write('\a')
                sys.stdout.flush()
                record = np.array(audio_chunk, dtype=np.int16)
                for sample_num in range(0, REC_LENGTH):
                    audio_chunk = self.audio_rec.read_chank()[1]
                    record = np.append(record, audio_chunk)
                rec_path = self.ctx["rec_saver"].save(record, REC_FILE_NAME)
                detector_server_response = self.ctx["cat_detector_client"].send_file(rec_path)
                print(f"Detection server cat detected: {detector_server_response.get('cat_detected')}")
                if detector_server_response.get("cat_detected"):
                    print("Cat detected")
                    au.play_cat_sound()

    def change(self, ctx):
        return StateStandby(ctx)


class StateMachine:
    def __init__(self, audio_rec, detector, rec_saver, cat_detector_client):
        self.context = {
            "audio_rec": audio_rec,
            "detector": detector,
            "rec_saver": rec_saver,
            "cat_detector_client": cat_detector_client
        }
        self.currentState = StateStandby(self.context)

    def change_state(self):
        self.currentState = self.currentState.change(self.context)

    def run(self):
        self.currentState.run()
