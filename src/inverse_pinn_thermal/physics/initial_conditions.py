"""Initial condition helpers."""

from __future__ import annotations

import numpy as np
import torch

from inverse_pinn_thermal.config import InitialConfig


def initial_temperature_numpy(
    x: np.ndarray,
    y: np.ndarray,
    cfg: InitialConfig,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> np.ndarray:
    """Evaluate T(x,y,0) on NumPy arrays (physical coordinates)."""
    if cfg.type == "constant":
        return np.full_like(x, cfg.value, dtype=np.float64)
    if cfg.type == "modal":
        return cfg.value * np.sin(np.pi * x / length_x) * np.sin(np.pi * y / length_y)
    if cfg.type == "analytic":
        # Same as modal; kept as an alias for clarity in configs.
        return cfg.value * np.sin(np.pi * x / length_x) * np.sin(np.pi * y / length_y)
    raise ValueError(f"Unknown initial condition type: {cfg.type}")


def initial_temperature_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: InitialConfig,
    *,
    normalized: bool = True,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> torch.Tensor:
    """Evaluate IC. If ``normalized``, (x,y) are in [0,1] non-dimensional space."""
    if cfg.type == "constant":
        return torch.full_like(x, cfg.value)
    # In normalized coordinates the modal IC is sin(π x*) sin(π y*).
    if normalized:
        return cfg.value * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return (
        cfg.value
        * torch.sin(torch.pi * x / length_x)
        * torch.sin(torch.pi * y / length_y)
    )
