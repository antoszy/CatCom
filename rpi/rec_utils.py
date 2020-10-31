import pyaudio
import numpy as np


class AudioRecorder:
    RESPEAKER_WIDTH = 2
    RESPEAKER_INDEX = 0  # refer to input device id

    def __init__(self, sample_rate=16000, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        print("Audio stream initialized")

    def __enter__(self):
        self.stream = self.p.open(rate=self.sample_rate,
                                  format=self.p.get_format_from_width(self.RESPEAKER_WIDTH),
                                  channels=self.channels,
                                  input=True,
                                  input_device_index=self.RESPEAKER_INDEX,)
        print("Audio stream opened")
        return self

    def __exit__(self,
                 exception_type,
                 exception_value,
                 exception_traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print(f"{exception_type} \n {exception_value} \n {exception_traceback}")
        print("Audio stream closed")

    def read_chank(self, chank_size=1024):
        data = self.stream.read(chank_size)
        data_arr = np.fromstring(data,dtype=np.int16)
        return [data_arr[start_idx::self.channels] for start_idx in range(0, self.channels)]

