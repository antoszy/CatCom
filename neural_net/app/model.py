import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio


class CatDetector:
    def __init__(self):
        self.yamnet_model = self._load_yamnet_model()
        self.class_names = self._load_class_mapping(self.yamnet_model)
        self.cat_class_index = self.class_names.index('Cat')

    def _load_yamnet_model():
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        return hub.load(yamnet_model_handle)

    def _load_class_mapping(yamnet_model):
        class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names =list(pd.read_csv(class_map_path)['display_name'])
        return class_names
    
    def detect_cat(self, wav_data):
        scores, embeddings, spectrogram = self.yamnet_model(wav_data)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.math.argmax(class_scores)
        inferred_class = self.class_names[top_class]
        print(f"{float(class_scores[76]):.3f}")


@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

    