import csv
from types import SimpleNamespace

from candidate_store import CandidateStore


def read_index(store):
    with open(store.index_path, newline="") as f:
        return list(csv.DictReader(f))


def test_saves_clip_and_appends_index(tmp_path):
    store = CandidateStore(tmp_path / "candidates")
    first = store.save(b"RIFF1", 0.71234, True, "1")
    second = store.save(b"RIFF2", 0.01, False, None)

    assert first != second
    assert first.read_bytes() == b"RIFF1" and second.read_bytes() == b"RIFF2"
    rows = read_index(store)
    assert [r["file"] for r in rows] == [first.name, second.name]
    assert rows[0]["cat_score"] == "0.7123" and rows[0]["detected"] == "1"
    assert rows[0]["svm"] == "1" and rows[1]["svm"] == ""


def fake_disk(used_fraction):
    total = 1000
    used = int(total * used_fraction)
    return lambda _path: SimpleNamespace(total=total, used=used, free=total - used)


def test_skips_saving_when_disk_too_full(tmp_path):
    store = CandidateStore(tmp_path / "candidates", disk_usage=fake_disk(0.91))
    assert store.save(b"RIFF", 0.5, True, "1") is None
    assert not store.index_path.exists()
    assert list((tmp_path / "candidates").glob("*.wav")) == []


def test_saves_at_exactly_the_disk_limit(tmp_path):
    store = CandidateStore(tmp_path / "candidates", disk_usage=fake_disk(0.90))
    assert store.save(b"RIFF", 0.5, True, "1") is not None
