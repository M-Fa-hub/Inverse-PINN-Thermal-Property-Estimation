"""Collocation-point sampling: uniform, LHS, and Sobol."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy.stats import qmc

SamplingMethod = Literal["uniform", "lhs", "sobol"]


def _unit_sample(n: int, d: int, method: SamplingMethod, seed: int) -> np.ndarray:
    """Sample n points in the unit hypercube [0, 1]^d."""
    if n <= 0:
        raise ValueError("n must be positive")
    if method == "uniform":
        rng = np.random.default_rng(seed)
        return rng.random((n, d))
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        return sampler.random(n)
    if method == "sobol":
        # Sobol prefers power-of-two sizes; pad then truncate.
        m = int(np.ceil(np.log2(max(n, 2))))
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        pts = sampler.random_base2(m)
        return pts[:n]
    raise ValueError(f"Unknown sampling method: {method}")


def sample_interior(
    n: int,
    method: SamplingMethod = "sobol",
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Interior collocation points with x,y,t ∈ (0,1), excluding boundaries."""
    pts = _unit_sample(n, 3, method, seed)
    # Shrink slightly away from boundaries for strict interior.
    pts = 1e-4 + (1.0 - 2e-4) * pts
    return _to_dict(pts, device)


def sample_initial(
    n: int,
    method: SamplingMethod = "sobol",
    seed: int = 1,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Initial-condition points: t = 0, x,y ∈ [0,1]."""
    pts = _unit_sample(n, 2, method, seed)
    t = np.zeros((n, 1), dtype=np.float64)
    full = np.concatenate([pts, t], axis=1)
    return _to_dict(full, device)


def sample_boundary(
    n: int,
    method: SamplingMethod = "sobol",
    seed: int = 2,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Boundary points on the four edges of the unit square, t ∈ [0,1].

    Returns also an integer ``edge`` id:
      0: x=0, 1: x=1, 2: y=0, 3: y=1
    """
    n_per = max(1, n // 4)
    remainder = n - 4 * n_per
    chunks: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    for edge_id in range(4):
        n_e = n_per + (1 if edge_id < remainder else 0)
        uv = _unit_sample(n_e, 2, method, seed + edge_id)
        u, t = uv[:, 0:1], uv[:, 1:2]
        if edge_id == 0:
            x = np.zeros_like(u)
            y = u
        elif edge_id == 1:
            x = np.ones_like(u)
            y = u
        elif edge_id == 2:
            x = u
            y = np.zeros_like(u)
        else:
            x = u
            y = np.ones_like(u)
        chunks.append(np.concatenate([x, y, t], axis=1))
        edges.append(np.full((n_e,), edge_id, dtype=np.int64))
    pts = np.concatenate(chunks, axis=0)
    edge = np.concatenate(edges, axis=0)
    out = _to_dict(pts, device)
    out["edge"] = torch.as_tensor(edge, device=device)
    return out


def sample_near_points(
    centers: np.ndarray,
    n: int,
    radius: float = 0.05,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Sample points in balls around residual hotspots (adaptive refinement)."""
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (m, 3)")
    rng = np.random.default_rng(seed)
    m = centers.shape[0]
    idx = rng.integers(0, m, size=n)
    noise = rng.normal(0.0, radius, size=(n, 3))
    pts = np.clip(centers[idx] + noise, 0.0, 1.0)
    return _to_dict(pts, device)


def _to_dict(pts: np.ndarray, device: torch.device | None) -> dict[str, torch.Tensor]:
    t = torch.as_tensor(pts, dtype=torch.float32, device=device)
    return {"x": t[:, 0:1], "y": t[:, 1:2], "t": t[:, 2:3]}
