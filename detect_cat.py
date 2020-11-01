from datetime import datetime
import os
from rpi import rec_utils
import utils.audio_utils as au
import numpy as np

DIRECTORY = "records_loud_90"
REC_LENGTH = 50

if __name__ == "__main__":
    if not os.path.exists(DIRECTORY):
        print(f"Creating directory {DIRECTORY}")
        os.makedirs(DIRECTORY)
        file_counter = 0
    else:
        file_counter = au.get_initial_file_no(DIRECTORY, "rec_a")

    with rec_utils.AudioRecorder() as audio_rec:
        while(1):
            audio_chank = audio_rec.read_chank()[1]
            if (au.check_threshold(audio_chank, norm_tsh=0.98)):
                print("Loud audio detected at:")
                print(datetime.now().time())
                record = np.array(audio_chank, dtype=np.int16)
                for sample_num in range(0, REC_LENGTH):
                    audio_chank = audio_rec.read_chank()[1]
                    record = np.append(record, audio_chank)
                au.save_wav(DIRECTORY+"/rec_a"+str(file_counter).zfill(5)+".wav",
                            record,
                            audio_rec.sample_rate)
                file_counter += 1

