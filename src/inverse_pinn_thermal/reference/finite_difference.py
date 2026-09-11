"""Self-contained 2D explicit finite-difference heat solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inverse_pinn_thermal.config import ExperimentSettings
from inverse_pinn_thermal.physics.initial_conditions import initial_temperature_numpy


@dataclass
class FDSolution:
    """Structured reference solution on a Cartesian grid."""

    x: np.ndarray  # (nx,)
    y: np.ndarray  # (ny,)
    t: np.ndarray  # (nt,)
    temperature: np.ndarray  # (nt, ny, nx)
    alpha: float

    def interpolate(self, xq: np.ndarray, yq: np.ndarray, tq: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of T at query points (physical coords)."""
        from scipy.interpolate import RegularGridInterpolator

        # RegularGridInterpolator expects (t, y, x) order matching temperature axes.
        interp = RegularGridInterpolator(
            (self.t, self.y, self.x),
            self.temperature,
            bounds_error=False,
            fill_value=None,
        )
        pts = np.stack([tq, yq, xq], axis=-1)
        return interp(pts)


def _cfl_number(alpha: float, dx: float, dy: float, dt: float) -> float:
    return alpha * dt * (1.0 / dx**2 + 1.0 / dy**2)


def solve_heat_equation_fd(
    length_x: float,
    length_y: float,
    final_time: float,
    alpha: float,
    nx: int,
    ny: int,
    nt: int,
    *,
    initial_type: str = "modal",
    initial_value: float = 1.0,
    boundary_type: str = "dirichlet_homogeneous",
    heat_source: float = 0.0,
) -> FDSolution:
    """Explicit FTCS finite-difference solver for 2D transient heat conduction.

    Supports homogeneous Dirichlet BCs (default benchmark). Raises if the CFL
    condition is violated: α Δt (1/Δx² + 1/Δy²) ≤ 1/2.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if boundary_type != "dirichlet_homogeneous":
        raise NotImplementedError(
            f"FD reference solver currently supports dirichlet_homogeneous only; "
            f"got {boundary_type}"
        )

    x = np.linspace(0.0, length_x, nx)
    y = np.linspace(0.0, length_y, ny)
    t = np.linspace(0.0, final_time, nt)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]

    cfl = _cfl_number(alpha, dx, dy, dt)
    if cfl > 0.5 + 1e-12:
        raise ValueError(
            f"Explicit FD CFL violated: CFL={cfl:.4f} > 0.5. "
            f"Increase nx/ny or nt, or reduce alpha/final_time."
        )

    xx, yy = np.meshgrid(x, y, indexing="xy")
    from inverse_pinn_thermal.config import InitialConfig

    ic_cfg = InitialConfig(type=initial_type, value=initial_value)  # type: ignore[arg-type]
    T = initial_temperature_numpy(xx, yy, ic_cfg, length_x, length_y)
    T[0, :] = 0.0
    T[-1, :] = 0.0
    T[:, 0] = 0.0
    T[:, -1] = 0.0

    history = np.zeros((nt, ny, nx), dtype=np.float64)
    history[0] = T

    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2
    q = heat_source

    for n in range(1, nt):
        Tn = T.copy()
        T[1:-1, 1:-1] = (
            Tn[1:-1, 1:-1]
            + rx * (Tn[1:-1, 2:] - 2.0 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2])
            + ry * (Tn[2:, 1:-1] - 2.0 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1])
            + dt * q
        )
        T[0, :] = 0.0
        T[-1, :] = 0.0
        T[:, 0] = 0.0
        T[:, -1] = 0.0
        history[n] = T

    return FDSolution(x=x, y=y, t=t, temperature=history, alpha=alpha)


def solve_from_config(cfg: ExperimentSettings) -> FDSolution:
    """Run the FD solver using experiment settings."""
    return solve_heat_equation_fd(
        length_x=cfg.geometry.length_x,
        length_y=cfg.geometry.length_y,
        final_time=cfg.geometry.final_time,
        alpha=cfg.physics.true_alpha,
        nx=cfg.reference.nx,
        ny=cfg.reference.ny,
        nt=cfg.reference.nt,
        initial_type=cfg.initial.type,
        initial_value=cfg.initial.value,
        boundary_type=cfg.boundary.type,
        heat_source=cfg.physics.heat_source,
    )


def analytical_modal_solution(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    alpha: float,
    length_x: float = 1.0,
    length_y: float = 1.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Exact single-mode solution for homogeneous Dirichlet BCs.

    T = A exp(-π² α (1/Lx² + 1/Ly²) t) sin(π x/Lx) sin(π y/Ly)
    """
    decay = np.pi**2 * alpha * (1.0 / length_x**2 + 1.0 / length_y**2)
    return (
        amplitude
        * np.exp(-decay * t)
        * np.sin(np.pi * x / length_x)
        * np.sin(np.pi * y / length_y)
    )


def verify_fd_against_analytical(
    *,
    alpha: float = 0.05,
    length_x: float = 1.0,
    length_y: float = 1.0,
    final_time: float = 1.0,
    nx: int = 41,
    ny: int = 41,
    nt: int = 401,
) -> dict[str, float]:
    """Compare FD solution to the analytic modal field; return error metrics."""
    sol = solve_heat_equation_fd(
        length_x, length_y, final_time, alpha, nx, ny, nt, initial_type="modal"
    )
    xx, yy = np.meshgrid(sol.x, sol.y, indexing="xy")
    err_sq_sum = 0.0
    abs_sum = 0.0
    exact_sq_sum = 0.0
    n_tot = 0
    for i, ti in enumerate(sol.t):
        exact = analytical_modal_solution(xx, yy, ti, alpha, length_x, length_y)
        err = sol.temperature[i] - exact
        err_sq_sum += float(np.sum(err**2))
        abs_sum += float(np.sum(np.abs(err)))
        exact_sq_sum += float(np.sum(exact**2))
        n_tot += err.size
    rmse = float(np.sqrt(err_sq_sum / n_tot))
    mae = float(abs_sum / n_tot)
    rel_l2 = float(np.sqrt(err_sq_sum) / (np.sqrt(exact_sq_sum) + 1e-15))
    return {"rmse": rmse, "mae": mae, "relative_l2": rel_l2}
