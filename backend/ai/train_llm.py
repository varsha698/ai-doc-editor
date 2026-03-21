"""Backward-compatible entry point ΓÇö implementation lives in ``ai/training/train_llm.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "training" / "train_llm.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
