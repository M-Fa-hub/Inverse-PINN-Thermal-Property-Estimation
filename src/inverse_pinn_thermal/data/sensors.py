"""Sparse sensor placement and measurement generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inverse_pinn_thermal.config import ExperimentSettings
from inverse_pinn_thermal.data.noise import add_gaussian_noise
from inverse_pinn_thermal.reference.finite_difference import FDSolution


def sensor_coordinates(
    n_sensors: int,
    length_x: float,
    length_y: float,
    *,
    placement: str = "grid",
    seed: int = 42,
    margin: float = 0.1,
) -> np.ndarray:
    """Return sensor (x, y) coordinates of shape (n_sensors, 2).

    Sensors are placed strictly inside the domain (away from Dirichlet walls).
    """
    if n_sensors <= 0:
        raise ValueError("n_sensors must be positive")
    x0, x1 = margin * length_x, (1.0 - margin) * length_x
    y0, y1 = margin * length_y, (1.0 - margin) * length_y

    if placement == "random":
        rng = np.random.default_rng(seed)
        xs = rng.uniform(x0, x1, size=n_sensors)
        ys = rng.uniform(y0, y1, size=n_sensors)
        return np.stack([xs, ys], axis=1)

    if placement == "grid":
        n_side = int(np.ceil(np.sqrt(n_sensors)))
        gx = np.linspace(x0, x1, n_side)
        gy = np.linspace(y0, y1, n_side)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        # Deterministic subset if grid is denser than requested.
        if pts.shape[0] > n_sensors:
            idx = np.linspace(0, pts.shape[0] - 1, n_sensors).astype(int)
            pts = pts[idx]
        return pts[:n_sensors]

    raise ValueError(f"Unknown placement: {placement}")


def generate_measurements(
    solution: FDSolution,
    cfg: ExperimentSettings,
) -> pd.DataFrame:
    """Interpolate the reference field at sparse sensors and add noise."""
    coords = sensor_coordinates(
        cfg.sensors.n_sensors,
        cfg.geometry.length_x,
        cfg.geometry.length_y,
        placement=cfg.sensors.placement,
        seed=cfg.sensors.random_seed,
    )
    times = np.linspace(0.0, cfg.geometry.final_time, cfg.sensors.n_time_samples)

    rows: list[dict[str, float]] = []
    clean_temps: list[float] = []
    for sx, sy in coords:
        for tt in times:
            clean = float(solution.interpolate(np.array([sx]), np.array([sy]), np.array([tt]))[0])
            clean_temps.append(clean)
            rows.append({"x": float(sx), "y": float(sy), "t": float(tt), "temperature_clean": clean})

    noisy = add_gaussian_noise(
        np.asarray(clean_temps, dtype=np.float64),
        cfg.measurement.noise_std,
        seed=cfg.experiment.seed,
    )
    df = pd.DataFrame(rows)
    df["temperature"] = noisy
    return df


def save_measurements(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_measurements(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Measurements not found: {path}")
    return pd.read_csv(path)
