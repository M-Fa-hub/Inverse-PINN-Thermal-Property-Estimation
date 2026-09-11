"""Lightweight end-to-end inverse estimation smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

from inverse_pinn_thermal.config import load_config
from inverse_pinn_thermal.data.sensors import generate_measurements
from inverse_pinn_thermal.evaluation.metrics import evaluate_model_on_solution, parameter_metrics
from inverse_pinn_thermal.geometry import RectangleDomain
from inverse_pinn_thermal.reference.finite_difference import solve_from_config
from inverse_pinn_thermal.training.trainer import InversePINNTrainer

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_e2e_inverse_smoke(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.experiment.output_dir = str(tmp_path)
    cfg.training.adam_epochs = 400
    cfg.training.lbfgs_enabled = True
    cfg.training.lbfgs_max_iter = 50
    cfg.training.print_every = 100
    cfg.training.early_stopping_patience = 400

    sol = solve_from_config(cfg)
    df = generate_measurements(sol, cfg)
    trainer = InversePINNTrainer(cfg, df)
    result = trainer.train(run_dir=tmp_path / "smoke_run")

    assert result.alpha_physical is not None
    pm = parameter_metrics(cfg.physics.true_alpha, result.alpha_physical)
    # Inverse recovery need not be perfect in a short smoke run, but should move
    # toward the truth relative to a naive far guess band.
    assert pm.relative_error < 0.5

    metrics, _, _ = evaluate_model_on_solution(
        trainer.model,
        sol,
        RectangleDomain.from_config(cfg.geometry),
        temperature_scale=cfg.scaling.temperature_scale,
        normalize_inputs=cfg.scaling.normalize_inputs,
        max_times=5,
    )
    assert metrics.relative_l2 < 0.35
