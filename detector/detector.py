import numpy as np
from scipy.fft import fft


class Detector:
    def __init__(self, clf):
        self.clf = clf

    @staticmethod
    def rec2feature(record):
        rec_fft = fft(record)
        return np.abs(rec_fft)

    def run_detection(self, record):
        feature = self.rec2feature(record)
        return self.clf.predict(feature)
