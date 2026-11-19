#!/bin/bash
# Start the cat detection server (EfficientAT). Run setup_server.sh first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec .venv/bin/uvicorn cat_detection_server:app --host 0.0.0.0 --port "${CATCOM_PORT:-8981}"
