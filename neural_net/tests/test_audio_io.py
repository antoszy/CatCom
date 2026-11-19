import io
import wave

import numpy as np
import pytest

from audio_io import InvalidAudioError, decode_wav


def make_wav(samples: np.ndarray, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(samples.dtype.itemsize)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def test_decodes_int16_mono_to_unit_range():
    samples = np.array([0, 32767, -32767, 16384], dtype=np.int16)
    waveform, sr = decode_wav(make_wav(samples))
    assert sr == 16000
    assert waveform.dtype == np.float32
    np.testing.assert_allclose(waveform, [0.0, 1.0, -1.0, 0.5], atol=1e-4)


def test_stereo_is_mixed_to_mono():
    interleaved = np.array([1000, 3000, -2000, 2000], dtype=np.int16)  # L,R,L,R
    waveform, _ = decode_wav(make_wav(interleaved, channels=2))
    np.testing.assert_allclose(waveform * 32767, [2000, 0], atol=1)


def test_rejects_garbage_and_empty_audio():
    with pytest.raises(InvalidAudioError):
        decode_wav(b"definitely not a wav file")
    with pytest.raises(InvalidAudioError):
        decode_wav(make_wav(np.array([], dtype=np.int16)))
