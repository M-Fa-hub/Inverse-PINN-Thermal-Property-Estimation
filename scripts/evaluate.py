"""Evaluate a trained run directory."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_pinn_thermal.cli import evaluate_main

if __name__ == "__main__":
    evaluate_main()
