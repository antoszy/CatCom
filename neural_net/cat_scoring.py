"""Turn YAMNet per-frame class scores into a single cat-vocalisation score.

Kept free of TensorFlow so it can be unit-tested without the model.
"""
from typing import Sequence, Tuple

import numpy as np

# AudioSet / YAMNet display names of cat vocalisations. "Purr" and "Hiss" are
# deliberately left out: we want to react to meowing, not to any cat sound.
CAT_VOCALISATION_CLASSES: Tuple[str, ...] = ("Cat", "Meow", "Caterwaul")


def resolve_class_indices(class_names: Sequence[str],
                          wanted: Sequence[str] = CAT_VOCALISATION_CLASSES) -> Tuple[int, ...]:
    """Map class display names to indices; fail loudly if the class map changed."""
    missing = [name for name in wanted if name not in class_names]
    if missing:
        raise ValueError(f"Classes not found in model class map: {missing}")
    return tuple(list(class_names).index(name) for name in wanted)


def cat_score(frame_scores: np.ndarray, class_indices: Sequence[int]) -> float:
    """Score of the most cat-like frame.

    Per frame we take the max over cat classes, then the max over frames, so a
    short meow inside a longer clip is not diluted by the surrounding silence
    (which a mean over frames would do).

    Args:
        frame_scores: array of shape (num_frames, num_classes) with scores in [0, 1].
        class_indices: columns of ``frame_scores`` that count as a cat vocalisation.
    """
    scores = np.asarray(frame_scores)
    if scores.ndim != 2:
        raise ValueError(f"frame_scores must be 2-D (frames, classes), got shape {scores.shape}")
    if scores.shape[0] == 0 or not class_indices:
        return 0.0
    return float(scores[:, list(class_indices)].max())
