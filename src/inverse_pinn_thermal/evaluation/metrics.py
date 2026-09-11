"""Evaluation metrics for temperature fields and parameter estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from inverse_pinn_thermal.geometry import RectangleDomain
from inverse_pinn_thermal.reference.finite_difference import FDSolution


@dataclass
class FieldMetrics:
    mae: float
    rmse: float
    relative_l2: float


@dataclass
class ParameterMetrics:
    true_value: float
    estimated_value: float
    absolute_error: float
    relative_error: float


def parameter_metrics(true_alpha: float, estimated_alpha: float) -> ParameterMetrics:
    abs_err = abs(estimated_alpha - true_alpha)
    rel_err = abs_err / abs(true_alpha) if true_alpha != 0 else float("inf")
    return ParameterMetrics(true_alpha, estimated_alpha, abs_err, rel_err)


def field_metrics(pred: np.ndarray, true: np.ndarray) -> FieldMetrics:
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    rel_l2 = float(np.linalg.norm(err.ravel()) / (np.linalg.norm(true.ravel()) + 1e-15))
    return FieldMetrics(mae=mae, rmse=rmse, relative_l2=rel_l2)


@torch.no_grad()
def predict_field(
    model: nn.Module,
    solution: FDSolution,
    domain: RectangleDomain,
    *,
    temperature_scale: float = 1.0,
    normalize_inputs: bool = True,
    device: torch.device | None = None,
    time_indices: list[int] | None = None,
) -> np.ndarray:
    """Predict temperature on the reference grid (subset of times optional)."""
    device = device or next(model.parameters()).device
    model.eval()
    t_idx = time_indices if time_indices is not None else list(range(len(solution.t)))
    preds = np.zeros((len(t_idx), len(solution.y), len(solution.x)), dtype=np.float64)
    xx, yy = np.meshgrid(solution.x, solution.y, indexing="xy")
    for i, ti in enumerate(t_idx):
        tt = np.full_like(xx, solution.t[ti])
        x_t = torch.tensor(xx.ravel(), dtype=torch.float32, device=device).view(-1, 1)
        y_t = torch.tensor(yy.ravel(), dtype=torch.float32, device=device).view(-1, 1)
        t_t = torch.tensor(tt.ravel(), dtype=torch.float32, device=device).view(-1, 1)
        if normalize_inputs:
            x_t, y_t, t_t = domain.normalize_coords(x_t, y_t, t_t)
        out = model(x_t, y_t, t_t).detach().cpu().numpy().reshape(xx.shape)
        preds[i] = out * temperature_scale
    return preds


def evaluate_model_on_solution(
    model: nn.Module,
    solution: FDSolution,
    domain: RectangleDomain,
    *,
    temperature_scale: float = 1.0,
    normalize_inputs: bool = True,
    max_times: int = 21,
) -> tuple[FieldMetrics, np.ndarray, np.ndarray]:
    """Compare model predictions against FD reference on a time subsample."""
    nt = len(solution.t)
    if nt <= max_times:
        t_idx = list(range(nt))
    else:
        t_idx = list(np.linspace(0, nt - 1, max_times).astype(int))
    pred = predict_field(
        model,
        solution,
        domain,
        temperature_scale=temperature_scale,
        normalize_inputs=normalize_inputs,
        time_indices=t_idx,
    )
    true = solution.temperature[t_idx]
    return field_metrics(pred, true), pred, true
