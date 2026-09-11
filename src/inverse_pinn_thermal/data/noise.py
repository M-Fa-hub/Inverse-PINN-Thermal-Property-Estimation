"""Synthetic Gaussian measurement noise."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    values: np.ndarray,
    noise_std: float,
    seed: int = 0,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise N(0, noise_std²). Returns a copy."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if noise_std == 0:
        return np.asarray(values.copy())
    rng = np.random.default_rng(seed)
    return np.asarray(values + rng.normal(0.0, noise_std, size=values.shape))
