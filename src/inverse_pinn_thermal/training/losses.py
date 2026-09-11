"""Composite PINN / baseline loss terms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from inverse_pinn_thermal.config import BoundaryConfig, InitialConfig, LossConfig
from inverse_pinn_thermal.physics.boundary_conditions import (
    DirichletBC,
    NeumannBC,
    RobinBC,
    normal_derivative,
)
from inverse_pinn_thermal.physics.heat_equation import heat_equation_residual
from inverse_pinn_thermal.physics.initial_conditions import initial_temperature_torch


@dataclass
class LossWeights:
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_ic: float = 10.0
    lambda_data: float = 100.0
    lambda_reg: float = 0.0

    @classmethod
    def from_config(cls, cfg: LossConfig) -> LossWeights:
        return cls(
            lambda_pde=cfg.lambda_pde,
            lambda_bc=cfg.lambda_bc,
            lambda_ic=cfg.lambda_ic,
            lambda_data=cfg.lambda_data,
            lambda_reg=cfg.lambda_reg,
        )


@dataclass
class LossBreakdown:
    total: torch.Tensor
    pde: torch.Tensor
    bc: torch.Tensor
    ic: torch.Tensor
    data: torch.Tensor
    reg: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "pde": float(self.pde.detach().cpu()),
            "bc": float(self.bc.detach().cpu()),
            "ic": float(self.ic.detach().cpu()),
            "data": float(self.data.detach().cpu()),
            "reg": float(self.reg.detach().cpu()),
        }


def _mse(a: torch.Tensor) -> torch.Tensor:
    return torch.mean(a**2)


class InversePINNLoss(nn.Module):
    """Assemble PDE / IC / BC / data / regularization losses."""

    def __init__(
        self,
        weights: LossWeights,
        boundary: BoundaryConfig,
        initial: InitialConfig,
        *,
        aspect_ratio_sq: float = 1.0,
        heat_source: float = 0.0,
        conductivity: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.boundary = boundary
        self.initial = initial
        self.aspect_ratio_sq = aspect_ratio_sq
        self.heat_source = heat_source
        self.conductivity = conductivity
        self.dirichlet = DirichletBC(boundary.dirichlet_value)
        self.neumann_insulated = NeumannBC(conductivity, heat_flux=0.0)
        self.robin = RobinBC(conductivity, boundary.convection_h, boundary.ambient_temperature)

    def pde_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        residual = heat_equation_residual(
            model,
            x,
            y,
            t,
            alpha,
            heat_source=self.heat_source,
            aspect_ratio_sq=self.aspect_ratio_sq,
        )
        return _mse(residual)

    def ic_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        pred = model(x, y, t)
        target = initial_temperature_torch(x, y, self.initial, normalized=True)
        return _mse(pred - target)

    def bc_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        edge: torch.Tensor,
    ) -> torch.Tensor:
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        pred = model(x, y, t)
        btype = self.boundary.type

        if btype == "dirichlet_homogeneous":
            return _mse(self.dirichlet.residual(pred))

        if btype == "insulated_sides_dirichlet_bottom":
            # edge 2 (y=0): Dirichlet; edges 0,1,3: insulated Neumann
            dtdn = normal_derivative(pred, x, y, edge)
            is_bottom = edge.view(-1, 1) == 2
            res_d = self.dirichlet.residual(pred)
            res_n = self.neumann_insulated.residual(dtdn)
            residual = torch.where(is_bottom, res_d, res_n)
            return _mse(residual)

        if btype == "convective_all":
            dtdn = normal_derivative(pred, x, y, edge)
            return _mse(self.robin.residual(pred, dtdn))

        raise ValueError(f"Unknown boundary type: {btype}")

    def data_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        temperature: torch.Tensor,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        pred = model(x, y, t)
        target = temperature / temperature_scale
        return _mse(pred - target)

    def forward(
        self,
        model: nn.Module,
        *,
        alpha: torch.Tensor,
        interior: dict[str, torch.Tensor],
        initial: dict[str, torch.Tensor],
        boundary: dict[str, torch.Tensor],
        data: dict[str, torch.Tensor],
        temperature_scale: float = 1.0,
        include_pde: bool = True,
        include_bc: bool = True,
        include_ic: bool = True,
        include_data: bool = True,
    ) -> LossBreakdown:
        device = next(model.parameters()).device
        zero = torch.tensor(0.0, device=device)

        pde = (
            self.pde_loss(model, interior["x"], interior["y"], interior["t"], alpha)
            if include_pde
            else zero
        )
        ic = (
            self.ic_loss(model, initial["x"], initial["y"], initial["t"])
            if include_ic
            else zero
        )
        bc = (
            self.bc_loss(model, boundary["x"], boundary["y"], boundary["t"], boundary["edge"])
            if include_bc
            else zero
        )
        data_l = (
            self.data_loss(
                model,
                data["x"],
                data["y"],
                data["t"],
                data["temperature"],
                temperature_scale=temperature_scale,
            )
            if include_data
            else zero
        )
        reg = zero

        total = (
            self.weights.lambda_pde * pde
            + self.weights.lambda_ic * ic
            + self.weights.lambda_bc * bc
            + self.weights.lambda_data * data_l
            + self.weights.lambda_reg * reg
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite total loss detected: {total.item()}. "
                "Check scaling, learning rate, and parameter initialization."
            )
        return LossBreakdown(total=total, pde=pde, bc=bc, ic=ic, data=data_l, reg=reg)
