"""Domain geometry helpers for the 2D rectangular solid."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from inverse_pinn_thermal.config import GeometryConfig


@dataclass(frozen=True)
class RectangleDomain:
    """Axis-aligned rectangle [0, Lx] × [0, Ly] with time [0, T]."""

    length_x: float
    length_y: float
    final_time: float

    @classmethod
    def from_config(cls, geometry: GeometryConfig) -> RectangleDomain:
        return cls(geometry.length_x, geometry.length_y, geometry.final_time)

    @property
    def aspect_ratio_sq(self) -> float:
        """(Lx / Ly)^2 factor appearing in the non-dimensional Laplacian."""
        return (self.length_x / self.length_y) ** 2

    def normalize_coords(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map physical (x,y,t) to non-dimensional (x*, y*, t*) in [0, 1]."""
        return x / self.length_x, y / self.length_y, t / self.final_time

    def denormalize_coords(
        self, x_star: torch.Tensor, y_star: torch.Tensor, t_star: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return x_star * self.length_x, y_star * self.length_y, t_star * self.final_time
