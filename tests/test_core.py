"""Unit and verification tests for inverse PINN thermal package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from inverse_pinn_thermal.config import load_config
from inverse_pinn_thermal.data.noise import add_gaussian_noise
from inverse_pinn_thermal.data.sensors import generate_measurements, sensor_coordinates
from inverse_pinn_thermal.inverse.parameters import PositiveParameter
from inverse_pinn_thermal.models.pinn import PINN
from inverse_pinn_thermal.physics.boundary_conditions import DirichletBC, NeumannBC, RobinBC
from inverse_pinn_thermal.physics.heat_equation import (
    ManufacturedSolutionModel,
    heat_equation_residual,
    manufactured_solution,
)
from inverse_pinn_thermal.reference.finite_difference import (
    solve_heat_equation_fd,
    verify_fd_against_analytical,
)
from inverse_pinn_thermal.sampling import sample_boundary, sample_initial, sample_interior
from inverse_pinn_thermal.training.checkpointing import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_config_validation_smoke() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    assert cfg.physics.true_alpha > 0
    assert cfg.alpha_nondimensional() == pytest.approx(
        cfg.physics.true_alpha * cfg.geometry.final_time / cfg.geometry.length_x**2
    )


def test_config_rejects_bad_dimension(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    raw["problem"]["dimension"] = 3
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_manufactured_solution_pde_residual_near_zero() -> None:
    alpha = 0.05
    model = ManufacturedSolutionModel(alpha)
    n = 64
    torch.manual_seed(0)
    x = torch.rand(n, 1, dtype=torch.float64) * 0.9 + 0.05
    y = torch.rand(n, 1, dtype=torch.float64) * 0.9 + 0.05
    t = torch.rand(n, 1, dtype=torch.float64) * 0.9 + 0.05
    # Model uses float32 by default ops; cast inputs
    x32, y32, t32 = x.float(), y.float(), t.float()
    residual = heat_equation_residual(model, x32, y32, t32, alpha)
    assert float(residual.abs().mean().detach()) < 1e-4


def test_autograd_first_and_second_derivatives() -> None:
    alpha = 0.1
    x = torch.tensor([[0.25]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[0.3]], dtype=torch.float32, requires_grad=True)
    t = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True)
    T = manufactured_solution(x, y, t, alpha)
    dT_dx = torch.autograd.grad(T, x, create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x, create_graph=True)[0]
    # Analytic: dT/dx = π cos(πx) sin(πy) exp(...); d2T/dx2 = -π² T
    assert d2T_dx2 is not None
    assert torch.allclose(d2T_dx2, -(torch.pi**2) * T, atol=1e-5)


def test_boundary_condition_modules() -> None:
    t_pred = torch.tensor([[1.0], [2.0]])
    assert torch.allclose(DirichletBC(0.0).residual(t_pred), t_pred)
    dtdn = torch.tensor([[0.0], [0.5]])
    assert torch.allclose(NeumannBC(1.0, 0.0).residual(dtdn), -dtdn)
    robin = RobinBC(1.0, 2.0, t_inf=0.0)
    res = robin.residual(t_pred, dtdn)
    assert res.shape == t_pred.shape


def test_parameter_positivity_softplus() -> None:
    p = PositiveParameter(1e-5, transform="softplus")
    val = p()
    assert float(val.detach()) > 0
    assert abs(float(val.detach()) - 1e-5) < 1e-8
    # Gradient flows to raw parameter
    val.backward()
    assert p.raw.grad is not None


def test_parameter_positivity_exp() -> None:
    p = PositiveParameter(0.05, transform="exp")
    assert abs(p.item() - 0.05) < 1e-6


def test_network_output_shape() -> None:
    model = PINN(hidden_layers=[32, 32], activation="tanh")
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    t = torch.randn(10, 1)
    out = model(x, y, t)
    assert out.shape == (10, 1)


def test_sensor_sampling_and_noise() -> None:
    coords = sensor_coordinates(9, 1.0, 1.0, placement="grid", seed=0)
    assert coords.shape == (9, 2)
    assert np.all(coords[:, 0] > 0) and np.all(coords[:, 0] < 1)
    clean = np.ones(20)
    noisy = add_gaussian_noise(clean, 0.1, seed=1)
    assert noisy.shape == clean.shape
    assert not np.allclose(noisy, clean)
    assert np.allclose(add_gaussian_noise(clean, 0.0), clean)


def test_sampling_methods() -> None:
    for method in ("uniform", "lhs", "sobol"):
        interior = sample_interior(32, method=method, seed=0)  # type: ignore[arg-type]
        assert interior["x"].shape == (32, 1)
        initial = sample_initial(16, method=method, seed=1)  # type: ignore[arg-type]
        assert torch.allclose(initial["t"], torch.zeros_like(initial["t"]))
        boundary = sample_boundary(40, method=method, seed=2)  # type: ignore[arg-type]
        assert "edge" in boundary
        assert boundary["edge"].numel() == 40


def test_fd_solver_against_analytical() -> None:
    metrics = verify_fd_against_analytical(alpha=0.05, final_time=1.0, nx=21, ny=21, nt=401)
    assert metrics["relative_l2"] < 0.05
    assert metrics["rmse"] < 0.02


def test_fd_cfl_guard() -> None:
    with pytest.raises(ValueError, match="CFL"):
        solve_heat_equation_fd(1.0, 1.0, 1.0, alpha=0.5, nx=11, ny=11, nt=5)


def test_measurements_from_fd() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    sol = solve_heat_equation_fd(
        cfg.geometry.length_x,
        cfg.geometry.length_y,
        cfg.geometry.final_time,
        cfg.physics.true_alpha,
        cfg.reference.nx,
        cfg.reference.ny,
        cfg.reference.nt,
    )
    df = generate_measurements(sol, cfg)
    assert set(["x", "y", "t", "temperature"]).issubset(df.columns)
    assert len(df) == cfg.sensors.n_sensors * cfg.sensors.n_time_samples


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = PINN(hidden_layers=[8, 8])
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        inverse_params=None,
        optimizer=None,
        epoch=3,
        seed=1,
        config={"a": 1},
        loss_history={"total": [1.0]},
        alpha_history=[0.1],
    )
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 3
    model2 = PINN(hidden_layers=[8, 8])
    model2.load_state_dict(loaded["model_state"])
