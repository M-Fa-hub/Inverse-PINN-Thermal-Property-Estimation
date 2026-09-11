"""Configuration loading and validation with Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ExperimentConfig(BaseModel):
    name: str = "inverse_alpha"
    seed: int = 42
    output_dir: str = "outputs"
    use_mlflow: bool = False


class ProblemConfig(BaseModel):
    dimension: int = 2
    transient: bool = True

    @field_validator("dimension")
    @classmethod
    def _dim_ok(cls, v: int) -> int:
        if v != 2:
            raise ValueError("Version 1 supports dimension=2 only.")
        return v


class GeometryConfig(BaseModel):
    length_x: float = Field(gt=0)
    length_y: float = Field(gt=0)
    final_time: float = Field(gt=0)


class PhysicsConfig(BaseModel):
    true_alpha: float = Field(gt=0)
    heat_source: float = 0.0
    conductivity: float = Field(default=1.0, gt=0)


class BoundaryConfig(BaseModel):
    type: Literal[
        "dirichlet_homogeneous",
        "insulated_sides_dirichlet_bottom",
        "convective_all",
    ] = "dirichlet_homogeneous"
    dirichlet_value: float = 0.0
    ambient_temperature: float = 0.0
    convection_h: float = Field(default=1.0, gt=0)


class InitialConfig(BaseModel):
    type: Literal["constant", "modal", "analytic"] = "modal"
    value: float = 1.0


class ScalingConfig(BaseModel):
    temperature_scale: float = Field(default=1.0, gt=0)
    normalize_inputs: bool = True


class InverseConfig(BaseModel):
    parameter: Literal["alpha", "k", "h"] = "alpha"
    initial_guess: float = Field(gt=0)
    transform: Literal["softplus", "exp"] = "softplus"


class NetworkConfig(BaseModel):
    hidden_layers: list[int] = Field(default_factory=lambda: [64, 64, 64, 64])
    activation: Literal["tanh", "relu", "gelu", "silu"] = "tanh"
    fourier_features: bool = False
    fourier_scale: float = 10.0
    fourier_features_dim: int = 64
    initialization: Literal["xavier", "kaiming"] = "xavier"

    @field_validator("hidden_layers")
    @classmethod
    def _layers_ok(cls, v: list[int]) -> list[int]:
        if not v or any(w <= 0 for w in v):
            raise ValueError("hidden_layers must be a non-empty list of positive ints.")
        return v


class SamplingConfig(BaseModel):
    n_interior: int = Field(gt=0)
    n_boundary: int = Field(gt=0)
    n_initial: int = Field(gt=0)
    method: Literal["uniform", "lhs", "sobol"] = "sobol"


class SensorsConfig(BaseModel):
    n_sensors: int = Field(gt=0)
    n_time_samples: int = Field(gt=0)
    placement: Literal["grid", "random"] = "grid"
    random_seed: int = 42


class MeasurementConfig(BaseModel):
    noise_std: float = Field(default=0.0, ge=0)


class ReferenceConfig(BaseModel):
    nx: int = Field(default=41, ge=5)
    ny: int = Field(default=41, ge=5)
    nt: int = Field(default=201, ge=5)
    scheme: Literal["explicit"] = "explicit"


class TrainingConfig(BaseModel):
    adam_epochs: int = Field(ge=0)
    learning_rate: float = Field(gt=0)
    lbfgs_enabled: bool = True
    lbfgs_max_iter: int = Field(default=500, ge=0)
    batch_interior: int | None = None
    scheduler: Literal["none", "cosine", "step"] = "cosine"
    early_stopping_patience: int = Field(default=1500, ge=1)
    grad_clip: float | None = 1.0
    checkpoint_every: int = Field(default=500, ge=1)
    resume_from_checkpoint: str | None = None
    print_every: int = Field(default=200, ge=1)
    device: str = "cpu"


class LossConfig(BaseModel):
    lambda_pde: float = Field(ge=0)
    lambda_bc: float = Field(ge=0)
    lambda_ic: float = Field(ge=0)
    lambda_data: float = Field(ge=0)
    lambda_reg: float = Field(default=0.0, ge=0)


class AdaptiveConfig(BaseModel):
    enabled: bool = False
    refine_every: int = Field(default=1000, ge=1)
    n_new_points: int = Field(default=500, ge=1)
    top_fraction: float = Field(default=0.2, gt=0, le=1)


class UncertaintyConfig(BaseModel):
    method: Literal["ensemble", "bootstrap"] = "ensemble"
    n_members: int = Field(default=3, ge=2)
    enabled: bool = False


class EvaluationConfig(BaseModel):
    grid_nx: int = Field(default=41, ge=5)
    grid_ny: int = Field(default=41, ge=5)
    grid_nt: int = Field(default=21, ge=2)


class ExperimentSettings(BaseModel):
    """Top-level validated configuration for the inverse PINN project."""

    experiment: ExperimentConfig
    problem: ProblemConfig
    geometry: GeometryConfig
    physics: PhysicsConfig
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    initial: InitialConfig = Field(default_factory=InitialConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    inverse: InverseConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    sampling: SamplingConfig
    sensors: SensorsConfig
    measurement: MeasurementConfig = Field(default_factory=MeasurementConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    training: TrainingConfig
    loss: LossConfig
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _check_cfl_hint(self) -> ExperimentSettings:
        # Soft check only; FD solver enforces CFL strictly.
        return self

    def alpha_nondimensional(self) -> float:
        """α_nd = α * t_final / Lx² for the scaled heat equation."""
        lx = self.geometry.length_x
        return self.physics.true_alpha * self.geometry.final_time / (lx**2)

    def initial_alpha_nondimensional(self) -> float:
        lx = self.geometry.length_x
        return self.inverse.initial_guess * self.geometry.final_time / (lx**2)

    def physical_from_nondim(self, alpha_nd: float) -> float:
        lx = self.geometry.length_x
        return alpha_nd * (lx**2) / self.geometry.final_time


def load_config(path: str | Path) -> ExperimentSettings:
    """Load and validate a YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping, got {type(raw)}")
    return ExperimentSettings.model_validate(raw)


def dump_config(settings: ExperimentSettings, path: str | Path) -> None:
    """Write a validated config to YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(settings.model_dump(), f, sort_keys=False)
