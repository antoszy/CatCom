#!/bin/bash
# Create .venv with server dependencies and download/verify model weights.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
.venv/bin/python -c "from cat_model import ensure_weights; print('Weights:', ensure_weights())"
