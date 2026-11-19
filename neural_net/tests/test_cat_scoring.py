import numpy as np
import pytest

from cat_scoring import CAT_VOCALISATION_CLASSES, cat_score, resolve_class_indices

CLASS_NAMES = ["Speech", "Dog", "Cat", "Purr", "Meow", "Hiss", "Caterwaul"]


def test_resolve_class_indices():
    assert resolve_class_indices(CLASS_NAMES) == (2, 4, 6)


def test_resolve_class_indices_fails_on_missing_class():
    with pytest.raises(ValueError, match="Meow"):
        resolve_class_indices(["Cat", "Caterwaul"])


def test_short_meow_is_not_diluted_by_other_frames():
    scores = np.zeros((4, len(CLASS_NAMES)))
    scores[2, CLASS_NAMES.index("Meow")] = 0.6  # one meow frame among silence
    assert cat_score(scores, resolve_class_indices(CLASS_NAMES)) == pytest.approx(0.6)


def test_non_meow_cat_sounds_are_ignored():
    scores = np.zeros((2, len(CLASS_NAMES)))
    scores[:, CLASS_NAMES.index("Purr")] = 0.9
    scores[:, CLASS_NAMES.index("Hiss")] = 0.9
    assert cat_score(scores, resolve_class_indices(CLASS_NAMES)) == 0.0


def test_empty_and_invalid_input():
    idx = resolve_class_indices(CLASS_NAMES)
    assert cat_score(np.zeros((0, len(CLASS_NAMES))), idx) == 0.0
    with pytest.raises(ValueError):
        cat_score(np.zeros(len(CLASS_NAMES)), idx)


def test_default_classes():
    assert CAT_VOCALISATION_CLASSES == ("Cat", "Meow", "Caterwaul")
