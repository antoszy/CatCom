from rpi import rec_utils
from rpi import pi_utils
import utils.audio_utils as au
from utils.state_machine import StateMachine
from utils.detector_client import CatDetectorClient

from detector.detector import Detector
import argparse
import pickle

DIRECTORY = "records_detector"
MODEL_DIR = "models/svc.pickle"
# PC running neural_net/cat_detection_server.py (EfficientAT).
CAT_DETECTOR_SERVER_URL = "http://192.168.0.4:8981/upload/"



def parse_args():
    parser = argparse.ArgumentParser(description="Kotofon: detect cat meows at the intercom.")
    parser.add_argument("--start-running", action="store_true",
                        help="start detecting immediately instead of waiting for the "
                             "button (used by the systemd service after boot)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rec_saver = au.RecSaver(DIRECTORY)
    with open(MODEL_DIR, "rb") as model_file:
        detector = Detector(pickle.load(model_file))
    audio_rec = rec_utils.AudioRecorder()
    cat_detector_client = CatDetectorClient(CAT_DETECTOR_SERVER_URL)
    state_machine = StateMachine(audio_rec, detector, rec_saver,
                                 cat_detector_client,
                                 start_running=args.start_running)
    button = pi_utils.Button(4)

    while 1:
        button_pressed = button.check_value()
        if button_pressed:
            state_machine.change_state()
        state_machine.run()
