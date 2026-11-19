"""Decode uploaded WAV bytes into a float waveform (stdlib only, no ffmpeg)."""
import io
import wave
from typing import Tuple

import numpy as np

_PCM_DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}


class InvalidAudioError(ValueError):
    """Uploaded data is not a WAV file we can decode."""


def decode_wav(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode PCM WAV bytes to a mono float32 waveform in [-1, 1].

    Returns:
        (waveform, sample_rate)

    Raises:
        InvalidAudioError: if the data is not an uncompressed 8/16/32-bit PCM WAV.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        raise InvalidAudioError(f"Not a readable PCM WAV file: {exc}") from exc

    dtype = _PCM_DTYPES.get(sample_width)
    if dtype is None:
        raise InvalidAudioError(f"Unsupported sample width: {sample_width} bytes")
    samples = np.frombuffer(frames, dtype=dtype)
    if samples.size == 0:
        raise InvalidAudioError("WAV file contains no audio")

    if dtype is np.uint8:  # 8-bit WAV is unsigned, centred at 128
        waveform = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        waveform = samples.astype(np.float32) / float(np.iinfo(dtype).max)
    if channels > 1:
        waveform = waveform[: waveform.size - waveform.size % channels]
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    return waveform, sample_rate
