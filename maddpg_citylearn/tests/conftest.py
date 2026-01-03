"""Test configuration: ensure project src is importable."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Bypass real model loading during tests
os.environ.setdefault("SKIP_MODEL_LOAD_FOR_TESTS", "1")
