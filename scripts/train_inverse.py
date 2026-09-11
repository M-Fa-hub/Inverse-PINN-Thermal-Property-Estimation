"""Train inverse PINN for thermal diffusivity estimation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_pinn_thermal.cli import train_inverse_main

if __name__ == "__main__":
    train_inverse_main()
