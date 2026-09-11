"""Physics-informed neural network for temperature field approximation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from inverse_pinn_thermal.config import NetworkConfig


def _get_activation(name: str) -> nn.Module:
    factories: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if name not in factories:
        raise ValueError(f"Unknown activation: {name}")
    return factories[name]()


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding for (x, y, t)."""

    def __init__(self, in_dim: int = 3, features: int = 64, scale: float = 10.0) -> None:
        super().__init__()
        b = torch.randn(in_dim, features) * scale
        self.register_buffer("B", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.B
        assert isinstance(b, torch.Tensor)
        proj = 2 * math.pi * x @ b
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINN(nn.Module):
    """MLP mapping (x, y, t) → T with optional Fourier features."""

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "tanh",
        *,
        fourier_features: bool = False,
        fourier_scale: float = 10.0,
        fourier_features_dim: int = 64,
        initialization: str = "xavier",
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [64, 64, 64, 64]
        self.use_fourier = fourier_features
        self.fourier: FourierFeatures | None = None
        if fourier_features:
            self.fourier = FourierFeatures(3, fourier_features_dim, fourier_scale)
            in_dim = 2 * fourier_features_dim
        else:
            in_dim = 3

        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            layers.append(_get_activation(activation))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights(initialization)

    @classmethod
    def from_config(cls, cfg: NetworkConfig) -> PINN:
        return cls(
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
            fourier_features=cfg.fourier_features,
            fourier_scale=cfg.fourier_scale,
            fourier_features_dim=cfg.fourier_features_dim,
            initialization=cfg.initialization,
        )

    def _init_weights(self, scheme: str) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {scheme}")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.cat([x, y, t], dim=-1)
        if self.fourier is not None:
            inp = self.fourier(inp)
        return self.net(inp)
