"""AudioSet labels and model width mapping (from upstream helpers/utils.py)."""
import csv
from pathlib import Path
from typing import List

LABELS_CSV = Path(__file__).with_name("class_labels_indices.csv")


def NAME_TO_WIDTH(name: str) -> float:  # noqa: N802 - upstream name, used by mn_model
    mn_map = {
        'mn01': 0.1, 'mn02': 0.2, 'mn04': 0.4, 'mn05': 0.5, 'mn06': 0.6, 'mn08': 0.8,
        'mn10': 1.0, 'mn12': 1.2, 'mn14': 1.4, 'mn16': 1.6, 'mn20': 2.0, 'mn30': 3.0,
        'mn40': 4.0,
    }
    dymn_map = {'dymn04': 0.4, 'dymn10': 1.0, 'dymn20': 2.0}
    if name.startswith('dymn'):
        return dymn_map.get(name[:6], 1.0)
    return mn_map.get(name[:4], 1.0)


def load_labels(path: Path = LABELS_CSV) -> List[str]:
    """Return AudioSet display names ordered by model output index."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [row["display_name"] for row in rows]
