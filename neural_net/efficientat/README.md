# Vendored EfficientAT (subset)

Source: https://github.com/fschmid56/EfficientAT, commit
`a425fdce92572e602a1d5634799bd9f1f2efa806` (2024-11-20), MIT license (`LICENSE`).

Copied files:

| Here | Upstream |
|---|---|
| `mn_model.py` | `models/mn/model.py` |
| `mn_utils.py` | `models/mn/utils.py` |
| `mn_block_types.py` | `models/mn/block_types.py` |
| `mn_attention_pooling.py` | `models/mn/attention_pooling.py` |
| `preprocess.py` | `models/preprocess.py` |
| `class_labels_indices.csv` | `metadata/class_labels_indices.csv` |
| `labels.py` | `NAME_TO_WIDTH` + label loading from `helpers/utils.py` |

Local changes: imports rewritten to package-relative ones, removed `print(m)`
(full architecture dump) from `get_model`, `labels.py` reads the CSV relative
to this file instead of the current working directory.

Pretrained weights are not stored here; `cat_model.py` downloads them from the
upstream GitHub release and verifies their SHA-256.
