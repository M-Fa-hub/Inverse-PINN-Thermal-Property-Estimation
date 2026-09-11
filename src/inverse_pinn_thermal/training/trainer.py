"""Adam + L-BFGS training loops for inverse PINNs and data-only baselines."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from inverse_pinn_thermal.config import ExperimentSettings
from inverse_pinn_thermal.geometry import RectangleDomain
from inverse_pinn_thermal.inverse.parameters import InverseParameters
from inverse_pinn_thermal.models.data_only import DataOnlyNetwork
from inverse_pinn_thermal.models.pinn import PINN
from inverse_pinn_thermal.sampling import (
    sample_boundary,
    sample_initial,
    sample_interior,
    sample_near_points,
)
from inverse_pinn_thermal.training.checkpointing import load_checkpoint, save_checkpoint
from inverse_pinn_thermal.training.losses import InversePINNLoss, LossWeights


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainResult:
    run_dir: Path
    alpha_physical: float | None
    alpha_nondim: float | None
    loss_history: dict[str, list[float]]
    alpha_history: list[float]
    best_total_loss: float
    epochs_run: int
    metrics: dict[str, Any] = field(default_factory=dict)


def _prepare_data_tensors(
    df: pd.DataFrame,
    domain: RectangleDomain,
    device: torch.device,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    x = torch.tensor(df["x"].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    t = torch.tensor(df["t"].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    temp = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    if normalize:
        x, y, t = domain.normalize_coords(x, y, t)
    return {"x": x, "y": y, "t": t, "temperature": temp}


def _maybe_mlflow_log(cfg: ExperimentSettings, params: dict[str, Any], metrics: dict[str, float]) -> None:
    if not cfg.experiment.use_mlflow:
        return
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow requested but not installed. pip install '.[mlflow]'") from exc
    with mlflow.start_run(run_name=cfg.experiment.name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for k, v in metrics.items():
            mlflow.log_metric(k, v)


class InversePINNTrainer:
    """Two-stage trainer: Adam warm-up then optional L-BFGS polish."""

    def __init__(self, cfg: ExperimentSettings, measurements: pd.DataFrame) -> None:
        self.cfg = cfg
        self.measurements = measurements
        self.device = torch.device(cfg.training.device)
        self.domain = RectangleDomain.from_config(cfg.geometry)
        set_seed(cfg.experiment.seed)

        self.model = PINN.from_config(cfg.network).to(self.device)
        # Optimize non-dimensional alpha: α_nd = α_phys * t_final / Lx²
        alpha_nd_init = cfg.initial_alpha_nondimensional()
        self.inverse_params = InverseParameters(
            estimate=cfg.inverse.parameter,
            alpha_init=alpha_nd_init,
            transform=cfg.inverse.transform,
        ).to(self.device)

        self.loss_fn = InversePINNLoss(
            LossWeights.from_config(cfg.loss),
            cfg.boundary,
            cfg.initial,
            aspect_ratio_sq=self.domain.aspect_ratio_sq,
            heat_source=cfg.physics.heat_source,
            conductivity=cfg.physics.conductivity,
        )

        self.interior = sample_interior(
            cfg.sampling.n_interior, cfg.sampling.method, cfg.experiment.seed, self.device
        )
        self.initial_pts = sample_initial(
            cfg.sampling.n_initial, cfg.sampling.method, cfg.experiment.seed + 1, self.device
        )
        self.boundary_pts = sample_boundary(
            cfg.sampling.n_boundary, cfg.sampling.method, cfg.experiment.seed + 2, self.device
        )
        self.data_pts = _prepare_data_tensors(
            measurements, self.domain, self.device, cfg.scaling.normalize_inputs
        )

        self.loss_history: dict[str, list[float]] = {
            k: [] for k in ("total", "pde", "bc", "ic", "data", "reg")
        }
        self.alpha_history: list[float] = []
        self.epoch = 0
        self.best_loss = float("inf")
        self.patience_counter = 0

    def _alpha_physical(self) -> float:
        alpha_nd = self.inverse_params.alpha_param.item()  # type: ignore[union-attr]
        return self.cfg.physical_from_nondim(alpha_nd)

    def _compute_loss(self) -> Any:
        alpha = self.inverse_params.alpha()
        return self.loss_fn(
            self.model,
            alpha=alpha,
            interior=self.interior,
            initial=self.initial_pts,
            boundary=self.boundary_pts,
            data=self.data_pts,
            temperature_scale=self.cfg.scaling.temperature_scale,
        )

    def _record(self, breakdown: Any) -> None:
        for k, v in breakdown.as_dict().items():
            self.loss_history[k].append(v)
        self.alpha_history.append(self._alpha_physical())

    def _adaptive_refine(self) -> None:
        if not self.cfg.adaptive.enabled:
            return
        with torch.enable_grad():
            alpha = self.inverse_params.alpha()
            from inverse_pinn_thermal.physics.heat_equation import heat_equation_residual

            residual = heat_equation_residual(
                self.model,
                self.interior["x"],
                self.interior["y"],
                self.interior["t"],
                alpha,
                heat_source=self.cfg.physics.heat_source,
                aspect_ratio_sq=self.domain.aspect_ratio_sq,
            ).detach()
        mag = residual.abs().view(-1)
        n_top = max(1, int(self.cfg.adaptive.top_fraction * mag.numel()))
        idx = torch.topk(mag, n_top).indices.cpu().numpy()
        centers = np.stack(
            [
                self.interior["x"].detach().cpu().numpy().ravel()[idx],
                self.interior["y"].detach().cpu().numpy().ravel()[idx],
                self.interior["t"].detach().cpu().numpy().ravel()[idx],
            ],
            axis=1,
        )
        new_pts = sample_near_points(
            centers,
            self.cfg.adaptive.n_new_points,
            seed=self.cfg.experiment.seed + self.epoch,
            device=self.device,
        )
        for key in ("x", "y", "t"):
            self.interior[key] = torch.cat([self.interior[key], new_pts[key]], dim=0)

    def train(self, run_dir: Path | None = None) -> TrainResult:
        cfg = self.cfg
        run_dir = run_dir or Path(cfg.experiment.output_dir) / cfg.experiment.name
        run_dir.mkdir(parents=True, exist_ok=True)

        if cfg.training.resume_from_checkpoint:
            self._resume(cfg.training.resume_from_checkpoint)

        params = list(self.model.parameters()) + list(self.inverse_params.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.training.learning_rate)
        scheduler: Any = None
        if cfg.training.scheduler == "cosine" and cfg.training.adam_epochs > 0:
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.adam_epochs)
        elif cfg.training.scheduler == "step" and cfg.training.adam_epochs > 0:
            scheduler = StepLR(optimizer, step_size=max(1, cfg.training.adam_epochs // 4), gamma=0.5)

        t0 = time.time()
        for epoch in range(self.epoch, cfg.training.adam_epochs):
            self.epoch = epoch
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            breakdown = self._compute_loss()
            breakdown.total.backward()
            if cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            self._record(breakdown)
            total = float(breakdown.total.detach())
            if total < self.best_loss - 1e-10:
                self.best_loss = total
                self.patience_counter = 0
                save_checkpoint(
                    run_dir / "best.pt",
                    model=self.model,
                    inverse_params=self.inverse_params,
                    optimizer=optimizer,
                    epoch=epoch,
                    seed=cfg.experiment.seed,
                    config=cfg.model_dump(),
                    loss_history=self.loss_history,
                    alpha_history=self.alpha_history,
                )
            else:
                self.patience_counter += 1

            if epoch % cfg.training.print_every == 0:
                print(
                    f"[Adam {epoch:5d}] loss={total:.4e} "
                    f"pde={breakdown.pde.item():.3e} data={breakdown.data.item():.3e} "
                    f"alpha={self._alpha_physical():.6e}"
                )

            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                save_checkpoint(
                    run_dir / f"epoch_{epoch+1:05d}.pt",
                    model=self.model,
                    inverse_params=self.inverse_params,
                    optimizer=optimizer,
                    epoch=epoch,
                    seed=cfg.experiment.seed,
                    config=cfg.model_dump(),
                    loss_history=self.loss_history,
                    alpha_history=self.alpha_history,
                )

            if cfg.adaptive.enabled and (epoch + 1) % cfg.adaptive.refine_every == 0:
                self._adaptive_refine()

            if self.patience_counter >= cfg.training.early_stopping_patience:
                print(f"Early stopping at Adam epoch {epoch}")
                break

        if cfg.training.lbfgs_enabled and cfg.training.lbfgs_max_iter > 0:
            self._run_lbfgs()

        # Reload best Adam weights if L-BFGS did not improve (keep final either way).
        save_checkpoint(
            run_dir / "final.pt",
            model=self.model,
            inverse_params=self.inverse_params,
            optimizer=None,
            epoch=self.epoch,
            seed=cfg.experiment.seed,
            config=cfg.model_dump(),
            loss_history=self.loss_history,
            alpha_history=self.alpha_history,
        )
        self._write_summaries(run_dir)

        alpha_nd = self.inverse_params.alpha_param.item()  # type: ignore[union-attr]
        alpha_phys = self.cfg.physical_from_nondim(alpha_nd)
        elapsed = time.time() - t0
        print(f"Training finished in {elapsed:.1f}s | alpha_est={alpha_phys:.6e}")

        _maybe_mlflow_log(
            cfg,
            {
                "true_alpha": cfg.physics.true_alpha,
                "initial_guess": cfg.inverse.initial_guess,
                "n_sensors": cfg.sensors.n_sensors,
                "noise_std": cfg.measurement.noise_std,
                "network": str(cfg.network.hidden_layers),
                "n_interior": cfg.sampling.n_interior,
            },
            {
                "alpha_estimated": alpha_phys,
                "best_total_loss": self.best_loss,
                "relative_alpha_error": abs(alpha_phys - cfg.physics.true_alpha)
                / cfg.physics.true_alpha,
            },
        )

        return TrainResult(
            run_dir=run_dir,
            alpha_physical=alpha_phys,
            alpha_nondim=alpha_nd,
            loss_history=self.loss_history,
            alpha_history=self.alpha_history,
            best_total_loss=self.best_loss,
            epochs_run=self.epoch + 1,
        )

    def _run_lbfgs(self) -> None:
        cfg = self.cfg
        params = list(self.model.parameters()) + list(self.inverse_params.parameters())
        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=cfg.training.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            breakdown = self._compute_loss()
            breakdown.total.backward()
            return breakdown.total

        print("Starting L-BFGS stage...")
        loss = optimizer.step(closure)
        breakdown = self._compute_loss()
        self._record(breakdown)
        loss_val = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        print(f"[L-BFGS] loss={loss_val:.4e} alpha={self._alpha_physical():.6e}")

    def _resume(self, path: str) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if ckpt.get("inverse_params_state"):
            self.inverse_params.load_state_dict(ckpt["inverse_params_state"])
        self.epoch = int(ckpt.get("epoch", 0)) + 1
        self.loss_history = ckpt.get("loss_history", self.loss_history)
        self.alpha_history = ckpt.get("alpha_history", [])
        print(f"Resumed from {path} at epoch {self.epoch}")

    def _write_summaries(self, run_dir: Path) -> None:
        alpha_phys = self._alpha_physical()
        summary = {
            "true_alpha": self.cfg.physics.true_alpha,
            "initial_guess": self.cfg.inverse.initial_guess,
            "estimated_alpha": alpha_phys,
            "absolute_error": abs(alpha_phys - self.cfg.physics.true_alpha),
            "relative_error": abs(alpha_phys - self.cfg.physics.true_alpha)
            / self.cfg.physics.true_alpha,
            "n_sensors": self.cfg.sensors.n_sensors,
            "noise_std": self.cfg.measurement.noise_std,
            "best_total_loss": self.best_loss,
            "epochs_run": self.epoch + 1,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        hist = pd.DataFrame(self.loss_history)
        hist["alpha"] = self.alpha_history + [None] * (len(hist) - len(self.alpha_history))
        hist.to_csv(run_dir / "history.csv", index=False)
        dump_path = run_dir / "config_resolved.yaml"
        from inverse_pinn_thermal.config import dump_config

        dump_config(self.cfg, dump_path)


class DataOnlyTrainer:
    """Baseline trainer: measurement loss only (no PDE/IC/BC)."""

    def __init__(self, cfg: ExperimentSettings, measurements: pd.DataFrame) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.domain = RectangleDomain.from_config(cfg.geometry)
        set_seed(cfg.experiment.seed)
        self.model = DataOnlyNetwork.from_config(cfg.network).to(self.device)
        self.loss_fn = InversePINNLoss(
            LossWeights.from_config(cfg.loss),
            cfg.boundary,
            cfg.initial,
            aspect_ratio_sq=self.domain.aspect_ratio_sq,
        )
        self.data_pts = _prepare_data_tensors(
            measurements, self.domain, self.device, cfg.scaling.normalize_inputs
        )
        self.loss_history: dict[str, list[float]] = {"total": [], "data": []}
        self.epoch = 0

    def train(self, run_dir: Path | None = None) -> TrainResult:
        cfg = self.cfg
        run_dir = run_dir or Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_baseline"
        run_dir.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.training.learning_rate)

        # Dummy alpha unused when PDE off
        alpha = torch.tensor(1.0, device=self.device)
        dummy_int = {
            "x": torch.zeros(1, 1, device=self.device),
            "y": torch.zeros(1, 1, device=self.device),
            "t": torch.zeros(1, 1, device=self.device),
        }
        dummy_edge = {"x": dummy_int["x"], "y": dummy_int["y"], "t": dummy_int["t"], "edge": torch.zeros(1, dtype=torch.long, device=self.device)}

        for epoch in range(cfg.training.adam_epochs):
            self.epoch = epoch
            optimizer.zero_grad(set_to_none=True)
            breakdown = self.loss_fn(
                self.model,
                alpha=alpha,
                interior=dummy_int,
                initial=dummy_int,
                boundary=dummy_edge,
                data=self.data_pts,
                temperature_scale=cfg.scaling.temperature_scale,
                include_pde=False,
                include_bc=False,
                include_ic=False,
                include_data=True,
            )
            breakdown.total.backward()
            if cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            self.loss_history["total"].append(float(breakdown.total.detach()))
            self.loss_history["data"].append(float(breakdown.data.detach()))
            if epoch % cfg.training.print_every == 0:
                print(f"[Baseline {epoch:5d}] data_loss={breakdown.data.item():.4e}")

        save_checkpoint(
            run_dir / "final.pt",
            model=self.model,
            inverse_params=None,
            optimizer=None,
            epoch=self.epoch,
            seed=cfg.experiment.seed,
            config=cfg.model_dump(),
            loss_history=self.loss_history,
            alpha_history=[],
        )
        summary = {
            "mode": "data_only",
            "n_sensors": cfg.sensors.n_sensors,
            "noise_std": cfg.measurement.noise_std,
            "final_data_loss": self.loss_history["data"][-1] if self.loss_history["data"] else None,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return TrainResult(
            run_dir=run_dir,
            alpha_physical=None,
            alpha_nondim=None,
            loss_history=self.loss_history,
            alpha_history=[],
            best_total_loss=min(self.loss_history["total"]) if self.loss_history["total"] else float("inf"),
            epochs_run=self.epoch + 1,
        )
