"""Checkpoint save / load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    inverse_params: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    seed: int,
    config: dict[str, Any],
    loss_history: dict[str, list[float]],
    alpha_history: list[float],
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "seed": seed,
        "model_state": model.state_dict(),
        "inverse_params_state": None if inverse_params is None else inverse_params.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "config": config,
        "loss_history": loss_history,
        "alpha_history": alpha_history,
        "extra": extra or {},
    }
    torch.save(payload, path)
    meta = {
        "epoch": epoch,
        "seed": seed,
        "alpha_last": alpha_history[-1] if alpha_history else None,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
