import pyaudio
import wave
import numpy as np
#import matplotlib.pyplot as plt
from datetime import datetime
import time
from pathlib import Path
 
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 2
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 0  # refer to input device id
CHUNK = 10 * RESPEAKER_RATE
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "output.wav"
WAVE_OUTPUT_FILENAME2 = "output2.wav"
record_data = []
record_data2 = []
DIRECTORY = "records_cat"
 
def get_initial_file_no(directory, file_name_base):
    filelist = Path(directory).glob(file_name_base+"*")
    num_gen = ( int(f.stem.lstrip(file_name_base)) for f in filelist)
    return max(num_gen) + 1
 
#for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
file_counter = 9
while(1):
    file_counter = get_initial_file_no(DIRECTORY, "rec_a")
    frames = [] 
    frames2 = [] 
    p = pyaudio.PyAudio()
    stream = p.open(
                rate=RESPEAKER_RATE,
                format=p.get_format_from_width(RESPEAKER_WIDTH),
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,)

    print(datetime.now().time())
    data = stream.read(CHUNK)
    a = np.fromstring(data,dtype=np.int16)[0::2]
    a2 = np.fromstring(data,dtype=np.int16)[1::2]
#    record_data.extend(a)
#    record_data2.extend(a2)
    frames.append(a.tostring())
    frames2.append(a2.tostring())

    wf = wave.open("records_cat/rec_a"+str(file_counter).zfill(5)+".wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    wf = wave.open("records_cat/rec_b"+str(file_counter).zfill(5)+".wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames2))
    wf.close()

    stream.stop_stream()
    stream.close()
    p.terminate()
    time.sleep(0)
    file_counter += 1
  
 
print("* done recording")
#plt.plot(record_data)
#plt.savefig("record.png")
 
stream.stop_stream()
stream.close()
p.terminate()
 
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames))
wf.close()

wf = wave.open(WAVE_OUTPUT_FILENAME2, 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames2))
wf.close()
