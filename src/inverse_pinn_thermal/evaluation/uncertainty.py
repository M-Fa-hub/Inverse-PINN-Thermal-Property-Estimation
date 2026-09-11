"""Empirical uncertainty via ensembles or bootstrap resampling."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from inverse_pinn_thermal.config import ExperimentSettings
from inverse_pinn_thermal.training.trainer import InversePINNTrainer, TrainResult


@dataclass
class UncertaintyResult:
    alpha_mean: float
    alpha_std: float
    alpha_samples: list[float]
    method: str
    note: str = (
        "Empirical uncertainty from independently trained models; "
        "not a calibrated Bayesian posterior."
    )


def run_ensemble_uncertainty(
    cfg: ExperimentSettings,
    measurements: pd.DataFrame,
    *,
    n_members: int | None = None,
    base_run_dir: Path | None = None,
) -> UncertaintyResult:
    """Train an ensemble of PINNs with different seeds."""
    n_members = n_members or cfg.uncertainty.n_members
    base_run_dir = base_run_dir or Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_ensemble"
    base_run_dir.mkdir(parents=True, exist_ok=True)

    samples: list[float] = []
    for i in range(n_members):
        member_cfg = cfg.model_copy(deep=True)
        member_cfg.experiment.seed = cfg.experiment.seed + 17 * (i + 1)
        member_cfg.experiment.name = f"{cfg.experiment.name}_ens{i}"
        # Keep training budget manageable for ensembles
        trainer = InversePINNTrainer(member_cfg, measurements)
        result: TrainResult = trainer.train(run_dir=base_run_dir / f"member_{i}")
        assert result.alpha_physical is not None
        samples.append(result.alpha_physical)

    arr = np.asarray(samples, dtype=np.float64)
    return UncertaintyResult(
        alpha_mean=float(arr.mean()),
        alpha_std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        alpha_samples=samples,
        method="ensemble",
    )


def run_bootstrap_uncertainty(
    cfg: ExperimentSettings,
    measurements: pd.DataFrame,
    *,
    n_members: int | None = None,
    base_run_dir: Path | None = None,
) -> UncertaintyResult:
    """Retrain on bootstrap-resampled measurement sets."""
    n_members = n_members or cfg.uncertainty.n_members
    base_run_dir = base_run_dir or Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_bootstrap"
    base_run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.experiment.seed)

    samples: list[float] = []
    n = len(measurements)
    for i in range(n_members):
        idx = rng.integers(0, n, size=n)
        boot_df = measurements.iloc[idx].reset_index(drop=True)
        member_cfg = copy.deepcopy(cfg)
        member_cfg.experiment.seed = cfg.experiment.seed + 31 * (i + 1)
        trainer = InversePINNTrainer(member_cfg, boot_df)
        result = trainer.train(run_dir=base_run_dir / f"member_{i}")
        assert result.alpha_physical is not None
        samples.append(result.alpha_physical)

    arr = np.asarray(samples, dtype=np.float64)
    return UncertaintyResult(
        alpha_mean=float(arr.mean()),
        alpha_std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        alpha_samples=samples,
        method="bootstrap",
    )
