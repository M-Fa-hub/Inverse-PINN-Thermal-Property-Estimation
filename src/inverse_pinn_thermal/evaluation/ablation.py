"""Ablation experiment runner (sensors / noise / baseline)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from inverse_pinn_thermal.config import ExperimentSettings, load_config
from inverse_pinn_thermal.data.sensors import generate_measurements, save_measurements
from inverse_pinn_thermal.evaluation.metrics import evaluate_model_on_solution, parameter_metrics
from inverse_pinn_thermal.geometry import RectangleDomain
from inverse_pinn_thermal.models.data_only import DataOnlyNetwork
from inverse_pinn_thermal.models.pinn import PINN
from inverse_pinn_thermal.reference.finite_difference import solve_from_config
from inverse_pinn_thermal.training.checkpointing import load_checkpoint
from inverse_pinn_thermal.training.trainer import DataOnlyTrainer, InversePINNTrainer


@dataclass
class AblationRow:
    experiment: str
    sensors: int
    noise: float
    true_alpha: float
    estimated_alpha: float | None
    relative_error: float | None
    field_rmse: float | None
    field_relative_l2: float | None
    mode: str


def _eval_checkpoint(
    cfg: ExperimentSettings,
    run_dir: Path,
    solution,
    *,
    baseline: bool = False,
) -> tuple[float | None, float | None, float | None]:
    ckpt_path = run_dir / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "best.pt"
    ckpt = load_checkpoint(ckpt_path)
    domain = RectangleDomain.from_config(cfg.geometry)
    if baseline:
        baseline_model = DataOnlyNetwork.from_config(cfg.network)
        baseline_model.load_state_dict(ckpt["model_state"])
        baseline_model.eval()
        metrics, _, _ = evaluate_model_on_solution(
            baseline_model,
            solution,
            domain,
            temperature_scale=cfg.scaling.temperature_scale,
            normalize_inputs=cfg.scaling.normalize_inputs,
            max_times=cfg.evaluation.grid_nt,
        )
        return None, metrics.rmse, metrics.relative_l2

    pinn_model = PINN.from_config(cfg.network)
    pinn_model.load_state_dict(ckpt["model_state"])
    pinn_model.eval()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    est = float(summary["estimated_alpha"])
    metrics, _, _ = evaluate_model_on_solution(
        pinn_model,
        solution,
        domain,
        temperature_scale=cfg.scaling.temperature_scale,
        normalize_inputs=cfg.scaling.normalize_inputs,
        max_times=cfg.evaluation.grid_nt,
    )
    return est, metrics.rmse, metrics.relative_l2


def run_ablation(cfg: ExperimentSettings, output_dir: Path | None = None) -> pd.DataFrame:
    """Run a compact ablation suite and return a results table.

    Experiments:
      1. data-only network
      2. PINN noiseless
      3. PINN moderate noise
      4. PINN fewer sensors
      5. PINN more sensors
    """
    output_dir = output_dir or Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared reference solution (true physics).
    solution = solve_from_config(cfg)
    torch.save(
        {"x": solution.x, "y": solution.y, "t": solution.t, "T": solution.temperature, "alpha": solution.alpha},
        output_dir / "reference.pt",
    )

    cases: list[tuple[str, int, float, str]] = [
        ("data_only", cfg.sensors.n_sensors, 0.0, "baseline"),
        ("pinn_clean", cfg.sensors.n_sensors, 0.0, "pinn"),
        ("pinn_noise", cfg.sensors.n_sensors, max(cfg.measurement.noise_std, 0.02), "pinn"),
        ("pinn_sparse", max(4, cfg.sensors.n_sensors // 2), 0.0, "pinn"),
        ("pinn_dense", cfg.sensors.n_sensors + 6, 0.0, "pinn"),
    ]

    rows: list[AblationRow] = []
    for name, n_sensors, noise, mode in cases:
        case_cfg = cfg.model_copy(deep=True)
        case_cfg.experiment.name = name
        case_cfg.sensors.n_sensors = n_sensors
        case_cfg.measurement.noise_std = noise
        case_dir = output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)

        df = generate_measurements(solution, case_cfg)
        save_measurements(df, case_dir / "measurements.csv")

        if mode == "baseline":
            trainer = DataOnlyTrainer(case_cfg, df)
            result = trainer.train(run_dir=case_dir)
            est, rmse, rel = _eval_checkpoint(case_cfg, result.run_dir, solution, baseline=True)
            rows.append(
                AblationRow(
                    experiment=name,
                    sensors=n_sensors,
                    noise=noise,
                    true_alpha=case_cfg.physics.true_alpha,
                    estimated_alpha=None,
                    relative_error=None,
                    field_rmse=rmse,
                    field_relative_l2=rel,
                    mode=mode,
                )
            )
        else:
            inv_trainer = InversePINNTrainer(case_cfg, df)
            result = inv_trainer.train(run_dir=case_dir)
            est, rmse, rel = _eval_checkpoint(case_cfg, result.run_dir, solution, baseline=False)
            if est is None:
                raise RuntimeError(f"Missing alpha estimate for ablation case {name}")
            pm = parameter_metrics(case_cfg.physics.true_alpha, est)
            rows.append(
                AblationRow(
                    experiment=name,
                    sensors=n_sensors,
                    noise=noise,
                    true_alpha=case_cfg.physics.true_alpha,
                    estimated_alpha=est,
                    relative_error=pm.relative_error,
                    field_rmse=rmse,
                    field_relative_l2=rel,
                    mode=mode,
                )
            )

    table = pd.DataFrame([asdict(r) for r in rows])
    table.to_csv(output_dir / "ablation_results.csv", index=False)
    (output_dir / "ablation_results.json").write_text(
        table.to_json(orient="records", indent=2), encoding="utf-8"
    )
    return table


def run_ablation_from_config(path: str | Path) -> pd.DataFrame:
    return run_ablation(load_config(path))
