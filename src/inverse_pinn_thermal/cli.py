"""Command-line entry points shared by scripts and console_scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inverse_pinn_thermal.config import load_config
from inverse_pinn_thermal.data.sensors import (
    generate_measurements,
    load_measurements,
    save_measurements,
)
from inverse_pinn_thermal.evaluation.ablation import run_ablation
from inverse_pinn_thermal.evaluation.metrics import evaluate_model_on_solution, parameter_metrics
from inverse_pinn_thermal.evaluation.uncertainty import (
    run_bootstrap_uncertainty,
    run_ensemble_uncertainty,
)
from inverse_pinn_thermal.geometry import RectangleDomain
from inverse_pinn_thermal.models.data_only import DataOnlyNetwork
from inverse_pinn_thermal.models.pinn import PINN
from inverse_pinn_thermal.reference.finite_difference import solve_from_config
from inverse_pinn_thermal.training.checkpointing import load_checkpoint
from inverse_pinn_thermal.training.trainer import DataOnlyTrainer, InversePINNTrainer
from inverse_pinn_thermal.visualization import plots


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")


def generate_reference_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate FD reference solution")
    _add_config_arg(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)
    sol = solve_from_config(cfg)
    out = run_dir / "reference.pt"
    torch.save(
        {"x": sol.x, "y": sol.y, "t": sol.t, "temperature": sol.temperature, "alpha": sol.alpha},
        out,
    )
    print(f"Saved reference solution to {out} shape={sol.temperature.shape}")


def generate_measurements_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate sparse sensor measurements")
    _add_config_arg(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ref_path = run_dir / "reference.pt"
    if ref_path.exists():
        blob = torch.load(ref_path, weights_only=False)
        from inverse_pinn_thermal.reference.finite_difference import FDSolution

        sol = FDSolution(
            x=blob["x"],
            y=blob["y"],
            t=blob["t"],
            temperature=blob["temperature"],
            alpha=float(blob["alpha"]),
        )
    else:
        sol = solve_from_config(cfg)
        torch.save(
            {"x": sol.x, "y": sol.y, "t": sol.t, "temperature": sol.temperature, "alpha": sol.alpha},
            ref_path,
        )
    df = generate_measurements(sol, cfg)
    path = save_measurements(df, run_dir / "measurements.csv")
    plots.plot_sensor_locations(
        df, run_dir / "figures" / "sensors.png", cfg.geometry.length_x, cfg.geometry.length_y
    )
    print(f"Saved {len(df)} measurements to {path}")


def train_inverse_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train inverse PINN")
    _add_config_arg(parser)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg.experiment.output_dir) / cfg.experiment.name
    meas_path = run_dir / "measurements.csv"
    if not meas_path.exists():
        generate_measurements_main(["--config", args.config])
    df = load_measurements(meas_path)
    trainer = InversePINNTrainer(cfg, df)
    result = trainer.train(run_dir=run_dir)
    print(json.dumps({"run_dir": str(result.run_dir), "alpha": result.alpha_physical}, indent=2))


def train_baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train data-only baseline")
    _add_config_arg(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_baseline"
    meas_path = Path(cfg.experiment.output_dir) / cfg.experiment.name / "measurements.csv"
    if not meas_path.exists():
        generate_measurements_main(["--config", args.config])
        meas_path = Path(cfg.experiment.output_dir) / cfg.experiment.name / "measurements.csv"
    df = load_measurements(meas_path)
    trainer = DataOnlyTrainer(cfg, df)
    result = trainer.train(run_dir=run_dir)
    print(f"Baseline finished: {result.run_dir}")


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run")
    parser.add_argument("--run", type=str, required=True, help="Run directory")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config_resolved.yaml") if (run_dir / "config_resolved.yaml").exists() else None
    if cfg is None:
        raise FileNotFoundError("config_resolved.yaml missing in run directory")

    ref_path = run_dir / "reference.pt"
    if not ref_path.exists():
        parent_ref = Path(cfg.experiment.output_dir) / cfg.experiment.name / "reference.pt"
        if parent_ref.exists():
            ref_path = parent_ref
        else:
            sol = solve_from_config(cfg)
            torch.save(
                {"x": sol.x, "y": sol.y, "t": sol.t, "temperature": sol.temperature, "alpha": sol.alpha},
                run_dir / "reference.pt",
            )
            ref_path = run_dir / "reference.pt"

    blob = torch.load(ref_path, weights_only=False)
    from inverse_pinn_thermal.reference.finite_difference import FDSolution

    sol = FDSolution(
        x=blob["x"],
        y=blob["y"],
        t=blob["t"],
        temperature=blob.get("temperature", blob.get("T")),
        alpha=float(blob["alpha"]),
    )
    domain = RectangleDomain.from_config(cfg.geometry)
    ckpt = load_checkpoint(run_dir / "final.pt")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model: PINN | DataOnlyNetwork
    if args.baseline or ckpt.get("inverse_params_state") is None:
        model = DataOnlyNetwork.from_config(cfg.network)
        model.load_state_dict(ckpt["model_state"])
        metrics, pred, true = evaluate_model_on_solution(
            model, sol, domain, temperature_scale=cfg.scaling.temperature_scale,
            normalize_inputs=cfg.scaling.normalize_inputs, max_times=cfg.evaluation.grid_nt,
        )
        report = {"field_mae": metrics.mae, "field_rmse": metrics.rmse, "field_relative_l2": metrics.relative_l2}
    else:
        model = PINN.from_config(cfg.network)
        model.load_state_dict(ckpt["model_state"])
        metrics, pred, true = evaluate_model_on_solution(
            model, sol, domain, temperature_scale=cfg.scaling.temperature_scale,
            normalize_inputs=cfg.scaling.normalize_inputs, max_times=cfg.evaluation.grid_nt,
        )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        pm = parameter_metrics(cfg.physics.true_alpha, float(summary["estimated_alpha"]))
        report = {
            "true_alpha": pm.true_value,
            "estimated_alpha": pm.estimated_value,
            "absolute_error": pm.absolute_error,
            "relative_error": pm.relative_error,
            "field_mae": metrics.mae,
            "field_rmse": metrics.rmse,
            "field_relative_l2": metrics.relative_l2,
        }
        hist_path = run_dir / "history.csv"
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            plots.plot_loss_history(hist, fig_dir / "losses.png")
            plots.plot_alpha_convergence(hist, cfg.physics.true_alpha, fig_dir / "alpha.png")

    mid = pred.shape[0] // 2
    plots.plot_temperature_slice(sol.x, sol.y, true[mid], fig_dir / "true_T.png", "True T")
    plots.plot_temperature_slice(sol.x, sol.y, pred[mid], fig_dir / "pinn_T.png", "PINN T")
    plots.plot_error_field(sol.x, sol.y, pred[mid] - true[mid], fig_dir / "error_T.png")

    meas_path = run_dir / "measurements.csv"
    if meas_path.exists():
        df = load_measurements(meas_path)
        plots.plot_sensor_locations(df, fig_dir / "sensors.png", cfg.geometry.length_x, cfg.geometry.length_y)
        # Time series at first sensor
        sensor = df[["x", "y"]].drop_duplicates().iloc[0]
        sub = df[(df["x"] == sensor["x"]) & (df["y"] == sensor["y"])].sort_values("t")
        t_vals = sub["t"].to_numpy()
        obs = sub["temperature"].to_numpy()
        ref = sol.interpolate(
            np.full_like(t_vals, sensor["x"]),
            np.full_like(t_vals, sensor["y"]),
            t_vals,
        )
        x_t = torch.tensor(np.full_like(t_vals, sensor["x"]), dtype=torch.float32).view(-1, 1)
        y_t = torch.tensor(np.full_like(t_vals, sensor["y"]), dtype=torch.float32).view(-1, 1)
        t_t = torch.tensor(t_vals, dtype=torch.float32).view(-1, 1)
        if cfg.scaling.normalize_inputs:
            x_t, y_t, t_t = domain.normalize_coords(x_t, y_t, t_t)
        with torch.no_grad():
            pred_ts = model(x_t, y_t, t_t).cpu().numpy().ravel() * cfg.scaling.temperature_scale
        plots.plot_sensor_timeseries(
            t_vals, ref, obs, pred_ts, fig_dir / "sensor_timeseries.png",
            title=f"Sensor ({sensor['x']:.2f}, {sensor['y']:.2f})",
        )
        plots.plot_pred_vs_measured(obs, pred_ts, fig_dir / "pred_vs_meas.png")

    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_ablation_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run ablation study")
    _add_config_arg(parser)
    parser.add_argument("--uncertainty", action="store_true", help="Also run ensemble uncertainty")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    table = run_ablation(cfg)
    out = Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_ablation"
    plots.plot_ablation_errors(table, out / "figures" / "ablation_errors.png")
    print(table.to_string(index=False))
    if args.uncertainty:
        meas = load_measurements(out / "pinn_clean" / "measurements.csv")
        unc_cfg = cfg.model_copy(deep=True)
        unc_cfg.uncertainty.enabled = True
        # Reduce budget for uncertainty members
        unc_cfg.training.adam_epochs = min(cfg.training.adam_epochs, 600)
        unc_cfg.training.lbfgs_max_iter = min(cfg.training.lbfgs_max_iter, 50)
        if cfg.uncertainty.method == "bootstrap":
            result = run_bootstrap_uncertainty(unc_cfg, meas, base_run_dir=out / "uncertainty")
        else:
            result = run_ensemble_uncertainty(unc_cfg, meas, base_run_dir=out / "uncertainty")
        payload = {
            "alpha_mean": result.alpha_mean,
            "alpha_std": result.alpha_std,
            "alpha_samples": result.alpha_samples,
            "method": result.method,
            "note": result.note,
        }
        (out / "uncertainty.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
