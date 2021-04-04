from rpi import rec_utils
from rpi import pi_utils
import utils.audio_utils as au
from utils.state_machine import StateMachine

from detector.detector import Detector
import pickle

DIRECTORY = "records_detector"
MODEL_DIR = "models/svc.pickle"


if __name__ == "__main__":
    rec_saver = au.RecSaver(DIRECTORY)
    with open(MODEL_DIR, "rb") as model_file:
        detector = Detector(pickle.load(model_file))
    audio_rec = rec_utils.AudioRecorder()
    state_machine = StateMachine(audio_rec, detector, rec_saver)
    button = pi_utils.Button(4)

    while 1:
        button_pressed = button.check_value()
        if button_pressed:
            state_machine.change_state()
        state_machine.run()
