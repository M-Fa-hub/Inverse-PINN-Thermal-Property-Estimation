"""Thin script wrappers around package CLI entry points."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running scripts without installation by adding src/ to path.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_pinn_thermal.cli import generate_reference_main

if __name__ == "__main__":
    generate_reference_main()
