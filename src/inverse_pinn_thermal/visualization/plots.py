"""Matplotlib visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temperature_slice(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(x, y, field, shading="auto", cmap="inferno")
    fig.colorbar(pcm, ax=ax, label="T")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    _savefig(path)


def plot_error_field(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    path: Path,
    title: str = "Absolute error",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(x, y, np.abs(err), shading="auto", cmap="magma")
    fig.colorbar(pcm, ax=ax, label="|error|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    _savefig(path)


def plot_sensor_locations(df: pd.DataFrame, path: Path, length_x: float = 1.0, length_y: float = 1.0) -> None:
    coords = df[["x", "y"]].drop_duplicates()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(coords["x"], coords["y"], c="crimson", s=40, zorder=3, label="sensors")
    ax.set_xlim(0, length_x)
    ax.set_ylim(0, length_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sensor locations")
    ax.set_aspect("equal")
    ax.legend()
    _savefig(path)


def plot_loss_history(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in ("total", "pde", "bc", "ic", "data"):
        if col in history.columns:
            ax.semilogy(history[col].to_numpy(), label=col)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training losses")
    ax.legend()
    _savefig(path)


def plot_alpha_convergence(history: pd.DataFrame, true_alpha: float, path: Path) -> None:
    if "alpha" not in history.columns:
        return
    alpha = history["alpha"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alpha, label="estimated α")
    ax.axhline(true_alpha, color="black", linestyle="--", label="true α")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("α")
    ax.set_title("Parameter convergence")
    ax.legend()
    _savefig(path)


def plot_pred_vs_measured(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Measured T")
    ax.set_ylabel("Predicted T")
    ax.set_title("Predicted vs measured")
    ax.set_aspect("equal")
    _savefig(path)


def plot_sensor_timeseries(
    t: np.ndarray,
    reference: np.ndarray,
    observed: np.ndarray | None,
    predicted: np.ndarray,
    path: Path,
    title: str = "Sensor time series",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, reference, "k-", label="reference")
    if observed is not None:
        ax.scatter(t, observed, c="tab:orange", s=18, label="noisy obs", zorder=3)
    ax.plot(t, predicted, "tab:blue", linestyle="--", label="PINN")
    ax.set_xlabel("t")
    ax.set_ylabel("T")
    ax.set_title(title)
    ax.legend()
    _savefig(path)


def plot_ablation_errors(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pinn = table[table["mode"] == "pinn"].copy()
    if not pinn.empty and "relative_error" in pinn.columns:
        axes[0].bar(pinn["experiment"], pinn["relative_error"])
        axes[0].set_ylabel("Relative α error")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].set_title("Parameter error")
    if "field_rmse" in table.columns:
        axes[1].bar(table["experiment"], table["field_rmse"])
        axes[1].set_ylabel("Field RMSE")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].set_title("Temperature RMSE")
    _savefig(path)
