"""Fixed-size buffer of the most recent audio chunks (pre-roll)."""
import math
from collections import deque
from typing import Deque

import numpy as np


def chunks_for_duration(duration_s: float, chunk_size: int, sample_rate: int) -> int:
    """Return the number of chunks needed to cover at least ``duration_s`` seconds."""
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    if chunk_size <= 0 or sample_rate <= 0:
        raise ValueError("chunk_size and sample_rate must be positive")
    return math.ceil(duration_s * sample_rate / chunk_size)


class ChunkRingBuffer:
    """Keeps the last ``max_chunks`` audio chunks; older ones are dropped."""

    def __init__(self, max_chunks: int):
        if max_chunks <= 0:
            raise ValueError(f"max_chunks must be positive, got {max_chunks}")
        self._chunks: Deque[np.ndarray] = deque(maxlen=max_chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def append(self, chunk: np.ndarray) -> None:
        self._chunks.append(chunk)

    def clear(self) -> None:
        self._chunks.clear()

    def to_array(self, dtype=np.int16) -> np.ndarray:
        """Concatenate buffered chunks, oldest first."""
        if not self._chunks:
            return np.array([], dtype=dtype)
        return np.concatenate(list(self._chunks)).astype(dtype, copy=False)
