from datetime import datetime
from rpi import rec_utils
import utils.audio_utils as au
from pathlib import Path

DIRECTORY = "records_loud"

file_counter = au.get_initial_file_no(DIRECTORY, "rec_a")
with rec_utils.AudioRecorder() as audio_rec:
    print(audio_rec)
    while(1):
        audio_chank = audio_rec.read_chank()[1]
        if (au.check_threshold(audio_chank, norm_tsh=0.475)):
            print("Loud audio detected at:")
            print(datetime.now().time())
            record = []
            record.extend(audio_chank)
            for sample_num in range(0,100):
                audio_chank = audio_rec.read_chank()[1]
                record.extend(audio_chank)
            au.save_wav(DIRECTORY+"/rec_a"+str(file_counter).zfill(5)+".wav",
                        record,
                        audio_rec.sample_rate)

