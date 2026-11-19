import sys
from pathlib import Path

# Server modules import each other as top-level modules (run from neural_net/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
