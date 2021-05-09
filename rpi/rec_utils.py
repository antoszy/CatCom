import pyaudio
import numpy as np


class AudioRecorder:
    RESPEAKER_WIDTH = 2
    RESPEAKER_INDEX = 0  # refer to input device id

    def __init__(self, sample_rate=16000, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.stream_running = False
        print("Audio stream initialized")

    def start_stream(self):
        self.stream = self.p.open(rate=self.sample_rate,
                                  format=self.p.get_format_from_width(self.RESPEAKER_WIDTH),
                                  channels=self.channels,
                                  input=True,
                                  input_device_index=self.RESPEAKER_INDEX,)
        self.stream_running = True
        print("Audio stream opened")

    def stop_stream(self):
        if self.stream_running:
            self.stream.stop_stream()
            self.stream.close()
            print("Audio stream closed")
        self.stream_running = False

    def __enter__(self):
        self.start_stream()
        return self

    def __exit__(self,
                 exception_type,
                 exception_value,
                 exception_traceback):
        self.stop_stream()
        print("Audio stream closed due to exception:")
        print(f"{exception_type} \n {exception_value} \n {exception_traceback}")

    def __del__(self):
        self.p.terminate()

    def read_chank(self, chank_size=1024):
        data = self.stream.read(chank_size, exception_on_overflow = False)
        data_arr = np.fromstring(data,dtype=np.int16)
        return [data_arr[start_idx::self.channels] for start_idx in range(0, self.channels)]

