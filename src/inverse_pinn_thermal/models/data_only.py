"""Data-only neural network baseline (no physics loss)."""

from __future__ import annotations

from inverse_pinn_thermal.config import NetworkConfig
from inverse_pinn_thermal.models.pinn import PINN


class DataOnlyNetwork(PINN):
    """Same architecture as the PINN; trained with measurement loss only."""

    @classmethod
    def from_config(cls, cfg: NetworkConfig) -> DataOnlyNetwork:
        return cls(
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
            fourier_features=cfg.fourier_features,
            fourier_scale=cfg.fourier_scale,
            fourier_features_dim=cfg.fourier_features_dim,
            initialization=cfg.initialization,
        )
