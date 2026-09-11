"""Trainable physical parameters with positivity constraints."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _inv_softplus(y: float) -> float:
    """Inverse softplus for a positive target y."""
    if y <= 0:
        raise ValueError("y must be positive")
    # softplus(x) = log(1 + exp(x)); inverse ≈ log(exp(y) - 1)
    if y > 20:
        return y
    return math.log(math.expm1(y))


class PositiveParameter(nn.Module):
    """Positive scalar parameter via softplus or exp transform."""

    def __init__(
        self,
        initial_value: float,
        *,
        transform: str = "softplus",
        name: str = "alpha",
    ) -> None:
        super().__init__()
        if initial_value <= 0:
            raise ValueError(f"{name} initial_value must be positive")
        self.name = name
        self.transform = transform
        if transform == "softplus":
            raw = _inv_softplus(initial_value)
        elif transform == "exp":
            raw = math.log(initial_value)
        else:
            raise ValueError(f"Unknown transform: {transform}")
        self.raw = nn.Parameter(torch.tensor(float(raw), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        if self.transform == "softplus":
            return torch.nn.functional.softplus(self.raw)
        return torch.exp(self.raw)

    @torch.no_grad()
    def item(self) -> float:
        return float(self.forward().detach().cpu())


class InverseParameters(nn.Module):
    """Container for unknown thermal properties (Version 1: alpha)."""

    def __init__(
        self,
        *,
        estimate: str = "alpha",
        alpha_init: float,
        transform: str = "softplus",
        k_init: float | None = None,
        h_init: float | None = None,
    ) -> None:
        super().__init__()
        self.estimate = estimate
        self.alpha_param: PositiveParameter | None = None
        self.k_param: PositiveParameter | None = None
        self.h_param: PositiveParameter | None = None

        if estimate == "alpha":
            self.alpha_param = PositiveParameter(alpha_init, transform=transform, name="alpha")
        elif estimate == "k":
            if k_init is None:
                raise ValueError("k_init required when estimating k")
            self.k_param = PositiveParameter(k_init, transform=transform, name="k")
        elif estimate == "h":
            if h_init is None:
                raise ValueError("h_init required when estimating h")
            self.h_param = PositiveParameter(h_init, transform=transform, name="h")
        else:
            raise ValueError(f"Unsupported estimate target: {estimate}")

    def alpha(self) -> torch.Tensor:
        if self.alpha_param is None:
            raise RuntimeError("alpha is not being estimated")
        return self.alpha_param()

    def current_values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.alpha_param is not None:
            out["alpha"] = self.alpha_param.item()
        if self.k_param is not None:
            out["k"] = self.k_param.item()
        if self.h_param is not None:
            out["h"] = self.h_param.item()
        return out
