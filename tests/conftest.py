import sys
from pathlib import Path

# Raspberry Pi side code lives in top-level packages (utils, rpi, detector).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
