import csv

import pytest
from fastapi.testclient import TestClient

from label_app import create_app
from label_store import LabelStore


@pytest.fixture
def recordings(tmp_path):
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    for name in ("rec_a00001.wav", "rec_a00002.wav"):
        (rec_dir / name).write_bytes(b"RIFF")
    (tmp_path / "secret.txt").write_text("nope")
    return rec_dir


def test_store_persists_and_removes_labels(tmp_path):
    path = tmp_path / "labels.csv"
    store = LabelStore(path)
    store.set("a.wav", "cat")
    store.set("b.wav", "not_cat")
    store.set("b.wav", None)
    reloaded = LabelStore(path)
    assert reloaded.all() == {"a.wav": "cat"}
    with open(path, newline="") as f:
        assert [row["file"] for row in csv.DictReader(f)] == ["a.wav"]


def test_store_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError):
        LabelStore(tmp_path / "l.csv").set("a.wav", "dog")


def test_api_lists_labels_and_saves(recordings, tmp_path):
    labels = tmp_path / "labels.csv"
    scores = {"rec_a00001.wav": 0.8}
    client = TestClient(create_app(recordings, labels, lambda p: scores.get(p.name)))
    items = client.get("/api/recordings").json()["items"]
    assert items[0] == {"file": "rec_a00001.wav", "score": 0.8, "label": None}

    assert client.put("/api/labels/rec_a00002.wav", json={"label": "unsure"}).status_code == 200
    assert LabelStore(labels).get("rec_a00002.wav") == "unsure"
    assert client.put("/api/labels/rec_a00002.wav", json={"label": "dog"}).status_code == 400


def test_only_listed_recordings_are_served(recordings, tmp_path):
    client = TestClient(create_app(recordings, tmp_path / "labels.csv"))
    assert client.get("/audio/rec_a00001.wav").status_code == 200
    assert client.get("/audio/..%2Fsecret.txt").status_code == 404
    assert client.put("/api/labels/unknown.wav", json={"label": "cat"}).status_code == 404


def test_index_page_is_served(recordings, tmp_path):
    client = TestClient(create_app(recordings, tmp_path / "labels.csv"))
    response = client.get("/")
    assert response.status_code == 200 and "labelowanie" in response.text


def test_new_recordings_appear_and_index_scores_are_used(recordings, tmp_path):
    client = TestClient(create_app(recordings, tmp_path / "labels.csv"))
    assert len(client.get("/api/recordings").json()["items"]) == 2

    (recordings / "cand_new.wav").write_bytes(b"RIFF")
    (recordings / "candidates.csv").write_text(
        "file,received_at,cat_score,detected,svm\ncand_new.wav,2026-11-19T10:00:00,0.61234,1,0\n")
    items = {i["file"]: i for i in client.get("/api/recordings").json()["items"]}
    assert items["cand_new.wav"]["score"] == 0.612
    assert client.get("/audio/cand_new.wav").status_code == 200
