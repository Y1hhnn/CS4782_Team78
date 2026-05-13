"""Pytest bootstrap: make `code/` importable so tests can `from tasks.* import ...`.

Allows `pytest code/tests/` (or `pytest`) from the repo root to work without
requiring `cd code` or setting PYTHONPATH manually.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
