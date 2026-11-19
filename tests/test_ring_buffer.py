import numpy as np
import pytest

from utils.ring_buffer import ChunkRingBuffer, chunks_for_duration


def test_chunks_for_duration_rounds_up():
    assert chunks_for_duration(1.0, 1024, 16000) == 16  # 15.625 -> 16
    assert chunks_for_duration(0.064, 1024, 16000) == 1
    assert chunks_for_duration(0.0, 1024, 16000) == 0


def test_chunks_for_duration_rejects_bad_input():
    with pytest.raises(ValueError):
        chunks_for_duration(-1.0, 1024, 16000)
    with pytest.raises(ValueError):
        chunks_for_duration(1.0, 0, 16000)


def test_keeps_only_newest_chunks_in_order():
    buf = ChunkRingBuffer(max_chunks=2)
    for value in (1, 2, 3):
        buf.append(np.full(3, value, dtype=np.int16))
    assert len(buf) == 2
    np.testing.assert_array_equal(buf.to_array(), [2, 2, 2, 3, 3, 3])
    assert buf.to_array().dtype == np.int16


def test_empty_and_clear():
    buf = ChunkRingBuffer(max_chunks=3)
    assert buf.to_array().size == 0
    buf.append(np.ones(4, dtype=np.int16))
    buf.clear()
    assert len(buf) == 0 and buf.to_array().size == 0


def test_rejects_non_positive_size():
    with pytest.raises(ValueError):
        ChunkRingBuffer(0)
