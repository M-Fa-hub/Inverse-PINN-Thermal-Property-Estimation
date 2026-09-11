"""Reusable thermal boundary-condition residual modules."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

EdgeId = Literal[0, 1, 2, 3]  # 0:x=0, 1:x=1, 2:y=0, 3:y=1


def outward_normal(edge: int) -> tuple[float, float]:
    """Unit outward normal for a unit-square edge."""
    if edge == 0:
        return (-1.0, 0.0)
    if edge == 1:
        return (1.0, 0.0)
    if edge == 2:
        return (0.0, -1.0)
    if edge == 3:
        return (0.0, 1.0)
    raise ValueError(f"Invalid edge id: {edge}")


class DirichletBC(nn.Module):
    """T = T_prescribed on a boundary."""

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = value

    def residual(self, t_pred: torch.Tensor) -> torch.Tensor:
        return t_pred - self.value


class NeumannBC(nn.Module):
    """-k ∂T/∂n = q_prescribed  ⇒  residual = -k dTdn - q."""

    def __init__(self, conductivity: float, heat_flux: float = 0.0) -> None:
        super().__init__()
        if conductivity <= 0:
            raise ValueError("conductivity must be positive")
        self.conductivity = conductivity
        self.heat_flux = heat_flux

    def residual(self, dtdn: torch.Tensor) -> torch.Tensor:
        return -self.conductivity * dtdn - self.heat_flux


class RobinBC(nn.Module):
    """Convective BC: -k ∂T/∂n = h (T - T_inf)."""

    def __init__(self, conductivity: float, h: float, t_inf: float = 0.0) -> None:
        super().__init__()
        if conductivity <= 0 or h <= 0:
            raise ValueError("conductivity and h must be positive")
        self.conductivity = conductivity
        self.h = h
        self.t_inf = t_inf

    def residual(self, t_pred: torch.Tensor, dtdn: torch.Tensor) -> torch.Tensor:
        return -self.conductivity * dtdn - self.h * (t_pred - self.t_inf)


def spatial_gradient(
    t_pred: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ∂T/∂x and ∂T/∂y via reverse-mode AD."""
    ones = torch.ones_like(t_pred)
    grads = torch.autograd.grad(
        t_pred,
        [x, y],
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    assert grads[0] is not None and grads[1] is not None
    return grads[0], grads[1]


def normal_derivative(
    t_pred: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    edge: torch.Tensor,
) -> torch.Tensor:
    """∂T/∂n for mixed edges using outward normals on the unit square."""
    dtdx, dtdy = spatial_gradient(t_pred, x, y)
    # Vectorized normal selection.
    nx = torch.zeros_like(dtdx)
    ny = torch.zeros_like(dtdy)
    nx = torch.where(edge.view(-1, 1) == 0, -torch.ones_like(nx), nx)
    nx = torch.where(edge.view(-1, 1) == 1, torch.ones_like(nx), nx)
    ny = torch.where(edge.view(-1, 1) == 2, -torch.ones_like(ny), ny)
    ny = torch.where(edge.view(-1, 1) == 3, torch.ones_like(ny), ny)
    return dtdx * nx + dtdy * ny
