from datetime import datetime
import os
from rpi import rec_utils
import utils.audio_utils as au
import numpy as np
from detector.detector import Detector
import pickle

DIRECTORY = "records_detector"
MODEL_DIR = "models/svc.pickle"
REC_LENGTH = 50


class RecSaver():
    def __init__(self, directory):
        if not os.path.exists(directory):
            print(f"RecSaver: Creating directory {directory}")
            os.makedirs(directory)
            self.file_counter = 0
        else:
            self.file_counter = au.get_initial_file_no(directory, "rec_a")
        self.directory = directory

    def save(self, record):
        au.save_wav(self.directory+"/rec_a"+str(self.file_counter).zfill(5)+".wav",
                    record,
                    audio_rec.sample_rate)
        self.file_counter += 1


if __name__ == "__main__":
    rec_saver = RecSaver(DIRECTORY)
    with open(MODEL_DIR, "rb") as model_file:
        detector = Detector(pickle.load(model_file))

    with rec_utils.AudioRecorder() as audio_rec:
        while(1):
            audio_chank = audio_rec.read_chank()[1]
            if (au.check_threshold(audio_chank, norm_tsh=0.50)):
                print("Loud audio detected at:")
                print(datetime.now().time())
                cat_detect = detector.run_detection(audio_chank)
                print(f"Effect of running detector = {cat_detect}")
                if cat_detect:
                    record = np.array(audio_chank, dtype=np.int16)
                    for sample_num in range(0, REC_LENGTH):
                        audio_chank = audio_rec.read_chank()[1]
                        record = np.append(record, audio_chank)
                    rec_saver.save(record)

