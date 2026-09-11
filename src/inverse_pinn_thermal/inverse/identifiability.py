"""Identifiability notes and simple correlation diagnostics."""

from __future__ import annotations

IDENTIFIABILITY_NOTES = """
Version 1 estimates a single unknown — thermal diffusivity α — with known
boundary/initial conditions. Estimating (α, k, h) jointly from sparse
temperature sensors is generally ill-posed: different (k, h) pairs can yield
similar boundary heat fluxes, and α is entangled with time scaling.

If multiple parameters are optimized, monitor pairwise correlations of the
learned trajectories and prefer fixing all but one property.
"""


def parameter_correlation(history: dict[str, list[float]]) -> dict[str, float]:
    """Pearson correlations between recorded parameter trajectories."""
    import numpy as np

    keys = [k for k, v in history.items() if len(v) > 1]
    out: dict[str, float] = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            xa = np.asarray(history[a], dtype=np.float64)
            xb = np.asarray(history[b], dtype=np.float64)
            if xa.std() < 1e-15 or xb.std() < 1e-15:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(xa, xb)[0, 1])
            out[f"{a}:{b}"] = corr
    return out
