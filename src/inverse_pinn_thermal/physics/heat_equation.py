"""Heat-equation residual via automatic differentiation."""

from __future__ import annotations

import torch
import torch.nn as nn


def heat_equation_residual(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    alpha: torch.Tensor | float,
    *,
    heat_source: float = 0.0,
    aspect_ratio_sq: float = 1.0,
) -> torch.Tensor:
    """Compute the non-dimensional heat residual.

    Formulation (non-dimensional coordinates x*, y*, t* ∈ [0,1]):

        ∂T*/∂t* - α_nd * (∂²T*/∂x*² + γ ∂²T*/∂y*²) - q* = 0

    where γ = (Lx/Ly)² and α_nd = α_physical * t_final / Lx².

    Derivatives are obtained with PyTorch autograd (no finite differences).
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    temperature = model(x, y, t)
    ones = torch.ones_like(temperature)

    dT_dt = torch.autograd.grad(
        temperature, t, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]
    dT_dx = torch.autograd.grad(
        temperature, x, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]
    dT_dy = torch.autograd.grad(
        temperature, y, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    assert dT_dt is not None and dT_dx is not None and dT_dy is not None

    d2T_dx2 = torch.autograd.grad(
        dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True, retain_graph=True
    )[0]
    d2T_dy2 = torch.autograd.grad(
        dT_dy, y, grad_outputs=torch.ones_like(dT_dy), create_graph=True, retain_graph=True
    )[0]
    assert d2T_dx2 is not None and d2T_dy2 is not None

    if not isinstance(alpha, torch.Tensor):
        alpha = torch.as_tensor(alpha, dtype=temperature.dtype, device=temperature.device)

    laplacian = d2T_dx2 + aspect_ratio_sq * d2T_dy2
    residual = dT_dt - alpha * laplacian - heat_source
    return residual


def manufactured_solution(
    x: torch.Tensor | float,
    y: torch.Tensor | float,
    t: torch.Tensor | float,
    alpha: float,
) -> torch.Tensor:
    """Analytic modal solution on the unit square with homogeneous Dirichlet BCs.

    T(x,y,t) = exp(-2 π² α t) sin(π x) sin(π y)

    Exactly satisfies ∂T/∂t = α ∇²T when (x,y,t) are non-dimensional and
    α is the matching non-dimensional diffusivity (γ = 1).
    """
    x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
    y_t = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.float64)
    t_t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=torch.float64)
    return (
        torch.exp(-2.0 * torch.pi**2 * alpha * t_t)
        * torch.sin(torch.pi * x_t)
        * torch.sin(torch.pi * y_t)
    )


class ManufacturedSolutionModel(nn.Module):
    """Wrapper exposing the manufactured field as a differentiable 'network'."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return manufactured_solution(x, y, t, self.alpha)
