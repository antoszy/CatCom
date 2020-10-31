from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wave


def draw_spectrogram(data, fs):
    f, t, Sxx = signal.spectrogram(data, fs, window=np.ones(1024))
    plt.figure(figsize=(23,7))
    plt.pcolormesh(t, f, Sxx, shading='gouraud', vmax=1e4)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()


def check_threshold(record, norm_tsh=0.75):
    tsh = np.iinfo(record.dtype).max * norm_tsh
    return tsh <= max(record)


def get_initial_file_no(directory, file_name_base):
    filelist = list(Path(directory).glob(file_name_base+"*"))
    if not filelist:
        return 0
    num_gen = ( int(f.stem.lstrip(file_name_base)) for f in filelist)
    return max(num_gen) + 1


def save_wav(file_path, record, sampe_rate=16000):
    frames = record.tostring()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(2)))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Record saved as {file_path}")

