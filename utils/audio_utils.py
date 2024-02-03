from pathlib import Path
import os
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np
import wave
import simpleaudio as sa


class RecSaver():
    def __init__(self, directory):
        if not os.path.exists(directory):
            print(f"RecSaver: Creating directory {directory}")
            os.makedirs(directory)
            self.file_counter = 0
        else:
            self.file_counter = get_initial_file_no(directory, "rec_a")
        self.directory = directory

    def save(self, record, file_name: str = None):
        if file_name is None:
            path = self.directory + "/rec_a"+str(self.file_counter).zfill(5)+".wav"
        else:
            path = self.directory + "/" + file_name
        save_wav(path,
                 record,
                 16000)
        # TODO: magic number above
        self.file_counter += 1
        return path


def draw_spectrogram(data, fs):
    f, t, Sxx = signal.spectrogram(data, fs, window=np.ones(1024))
    plt.figure(figsize=(23, 7))
    plt.pcolormesh(t, f, Sxx, shading='gouraud', vmax=1e4)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()


def check_threshold(record, norm_tsh=0.82):
    tsh = np.iinfo(record.dtype).max * norm_tsh
    return tsh <= max(record)


def get_initial_file_no(directory, file_name_base):
    filelist = list(Path(directory).glob(file_name_base+"*"))
    if not filelist:
        return 0
    num_gen = (int(f.stem.lstrip(file_name_base)) for f in filelist)
    return max(num_gen) + 1


def save_wav(file_path, record, sample_rate=16000):
    frames = record.tostring()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(frames)
    wf.close()
    print(f"Record saved as {file_path}")


def play_cat_sound():
    filename = 'records_detector/rec_a00515.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing


def play_a_sound():
    filename = 'sounds/a.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing


def play_d_sound():
    filename = 'sounds/d.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing
