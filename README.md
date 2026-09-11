# Inverse Physics-Informed Neural Network for Thermal Property Estimation

**Estimate unknown thermal diffusivity from sparse, noisy temperature measurements while enforcing the 2D transient heat equation.**

This is my scientific machine-learning portfolio project. I defined the research idea, problem formulation, validation plan, and engineering requirements. The implementation was written with the help of an AI coding assistant, then reviewed, tested, and iterated by me.

The repository is self-contained: inverse PINNs, PDE-constrained optimization, automatic differentiation, synthetic sensor design, uncertainty estimation, baseline comparison, and reproducible Python packaging — without depending on external experimental datasets.

| Aspect | Version 1 choice |
| ------ | ---------------- |
| PDE | 2D transient heat conduction |
| Unknown | Thermal diffusivity \(\alpha\) |
| Data | Synthetic sparse sensors + optional Gaussian noise |
| Reference | Built-in finite-difference solver (CFL-checked) |
| Baseline | Data-only neural network (no physics loss) |
| Stack | PyTorch · NumPy · SciPy · Pydantic · matplotlib · pytest |

---

## Table of contents

1. [Authorship & AI assistance](#1-authorship--ai-assistance)
2. [Scientific motivation](#2-scientific-motivation)
3. [Inverse PINN concept](#3-inverse-pinn-concept)
4. [Governing heat equation](#4-governing-heat-equation)
5. [Physical parameters & identifiability](#5-physical-parameters--identifiability)
6. [Benchmark problem setup](#6-benchmark-problem-setup)
7. [Synthetic reference & sensors](#7-synthetic-reference--sensors)
8. [Network architecture](#8-network-architecture)
9. [Inverse parameter learning](#9-inverse-parameter-learning)
10. [Loss formulation](#10-loss-formulation)
11. [Scaling & non-dimensionalization](#11-scaling--non-dimensionalization)
12. [Training strategy](#12-training-strategy)
13. [Evaluation metrics](#13-evaluation-metrics)
14. [Baseline & ablation study](#14-baseline--ablation-study)
15. [Uncertainty estimation](#15-uncertainty-estimation)
16. [Results](#16-results)
17. [Project structure](#17-project-structure)
18. [Installation](#18-installation)
19. [Configuration](#19-configuration)
20. [Exact run commands](#20-exact-run-commands)
21. [Outputs & artifacts](#21-outputs--artifacts)
22. [Testing & quality](#22-testing--quality)
23. [Reproducibility](#23-reproducibility)
24. [Limitations](#24-limitations)
25. [Next improvements](#25-next-improvements)
26. [License](#26-license)

---

## 1. Authorship & AI assistance

| Role | Contribution |
| ---- | ------------ |
| **Author** | Project idea, scientific scope, inverse-problem design, architecture choices, validation criteria, result interpretation, and final ownership of the work |
| **AI coding assistant** | Helped implement and refactor code, tests, configs, and documentation under my direction |

I remain responsible for the correctness, claims, and presentation of this project. AI assistance accelerated coding; it did not replace scientific judgment or project ownership.

---

## 2. Scientific motivation

Thermal properties of solids — diffusivity, conductivity, convection coefficients — are often **not measured directly**. Instead they are inferred from temperature histories recorded by a limited number of sensors. This is a classic **inverse heat-conduction problem**:

- The forward map (property → temperature field) is well-posed under standard assumptions.
- The inverse map (sparse temperatures → property) is typically ill-conditioned, noise-sensitive, and may be non-unique if too many parameters are free.

Classical workflows couple a numerical PDE solver to a nonlinear optimizer (e.g. least squares). **Physics-Informed Neural Networks (PINNs)** provide an alternative formulation:

1. A neural network \(T_\theta(x,y,t)\) approximates the temperature field.
2. Automatic differentiation evaluates the heat-equation residual everywhere in the domain.
3. Unknown material parameters (here \(\alpha\)) become trainable scalars.
4. Sparse measurements enter as a data-matching term.

The result is a single PDE-constrained learning problem that reconstructs the field **and** estimates the property.

---

## 3. Inverse PINN concept

```mermaid
flowchart TB
  subgraph Data Generation
    A[Finite-difference<br/>reference solver] --> B[Sparse sensor<br/>sampling]
    B --> C[Optional Gaussian<br/>measurement noise]
  end

  subgraph Inverse PINN
    D[Collocation points<br/>interior / IC / BC] --> F[Composite loss]
    C --> F
    E[MLP Tθ(x,y,t)<br/>+ trainable α] --> F
    F --> G[Adam warm-up]
    G --> H[L-BFGS polish]
  end

  subgraph Evaluation
    H --> I[Estimated α]
    H --> J[Reconstructed T field]
    I --> K[Parameter error<br/>& uncertainty]
    J --> L[MAE / RMSE / relative L2<br/>vs reference]
    M[Data-only baseline] --> L
  end
```

**What is learned**

| Output | Description |
| ------ | ----------- |
| \(T_\theta(x,y,t)\) | Continuous temperature field reconstruction |
| \(\alpha\) | Thermal diffusivity (primary unknown) |

**What is enforced**

| Constraint | Role |
| ---------- | ---- |
| Heat equation residual | Physics consistency in the interior |
| Initial condition | Correct state at \(t=0\) |
| Boundary conditions | Dirichlet / Neumann / Robin on walls |
| Sensor measurements | Match sparse observations |

The reference FD solution is used **only** to generate measurements and to validate predictions. It is **not** embedded inside the PINN loss except through the deliberately selected sparse sensor points.

---

## 4. Governing heat equation

### Physical form

\[
\frac{\partial T}{\partial t}
= \alpha \left(
\frac{\partial^2 T}{\partial x^2}
+ \frac{\partial^2 T}{\partial y^2}
\right)
+ q
\]

Equivalently, the residual form used in the PDE loss:

\[
\mathcal{R}(x,y,t)
=
\frac{\partial T}{\partial t}
- \alpha \nabla^2 T
- q
= 0
\]

All spatial and temporal derivatives inside the PINN are obtained with **PyTorch reverse-mode automatic differentiation**. Finite differences are **not** used for PINN PDE derivatives.

### Manufactured / analytic verification field

For the unit square with homogeneous Dirichlet boundaries, the single-mode solution

\[
T(x,y,t)
=
\mathrm{e}^{-2\pi^2 \alpha t}
\sin(\pi x)\sin(\pi y)
\]

satisfies the heat equation exactly. This manufactured solution is used to:

- verify the autograd residual implementation independently of training;
- verify the finite-difference reference solver against a known analytic field.

---

## 5. Physical parameters & identifiability

| Symbol | Name | Version 1 role |
| ------ | ---- | -------------- |
| \(\alpha\) | Thermal diffusivity | **Unknown — estimated** |
| \(k\) | Thermal conductivity | Known (used in Neumann/Robin BC modules) |
| \(h\) | Convective heat-transfer coefficient | Supported in BC API; not co-estimated in primary benchmark |
| \(q\) | Volumetric heat source (normalized) | Optional; default \(0\) |

### Why estimate only \(\alpha\) in Version 1?

Jointly estimating \((\alpha, k, h)\) from sparse temperature sensors is often **ill-posed**:

- Different \((k,h)\) pairs can produce similar boundary heat fluxes (Robin ambiguity).
- Diffusivity is entangled with time scaling; poor non-dimensionalization amplifies this.
- Sparse sensors provide limited information about spatial gradients needed to separate conductivity from diffusivity.

Version 1 therefore uses:

> **one unknown property + known boundary/initial conditions**

as the primary identifiable benchmark. See `src/inverse_pinn_thermal/inverse/identifiability.py`.

---

## 6. Benchmark problem setup

### Geometry & time

- Domain: rectangle \([0, L_x] \times [0, L_y]\)
- Time horizon: \(t \in [0, t_f]\)
- Default full benchmark: \(L_x = L_y = 1\), \(t_f = 100\), physical \(\alpha = 10^{-5}\)
- Fast smoke benchmark: \(t_f = 1\), \(\alpha = 0.05\) (non-dimensional-friendly)

### Initial condition

\[
T(x,y,0)
=
\sin\left(\frac{\pi x}{L_x}\right)
\sin\left(\frac{\pi y}{L_y}\right)
\]

Constant and analytic aliases are also configurable.

### Boundary conditions (reusable modules)

| Type | Mathematical form | Module |
| ---- | ----------------- | ------ |
| Dirichlet | \(T = T_b\) | `DirichletBC` |
| Neumann | \(-k\,\partial T/\partial n = q_b\) | `NeumannBC` |
| Robin / convective | \(-k\,\partial T/\partial n = h(T - T_\infty)\) | `RobinBC` |

**Default benchmark BC:** homogeneous Dirichlet on all four edges (\(T=0\)), compatible with the modal manufactured solution.

Alternative config options:

- `insulated_sides_dirichlet_bottom`
- `convective_all`

---

## 7. Synthetic reference & sensors

### Reference solver

A self-contained **explicit FTCS finite-difference** solver generates the ground-truth temperature field.

- Grid: configurable `nx`, `ny`, `nt`
- Hard **CFL guard**: \(\alpha\Delta t\,(1/\Delta x^2 + 1/\Delta y^2) \le 1/2\)
- Verified against the analytic modal solution before being used as a PINN judge

### Sparse sensors

Sensors are placed strictly inside the domain (away from Dirichlet walls):

| Setting | Description |
| ------- | ----------- |
| `n_sensors` | Number of spatial sensor locations |
| `n_time_samples` | Temporal samples per sensor |
| `placement` | `grid` or `random` |
| `noise_std` | i.i.d. Gaussian noise std on temperature |

Exported CSV schema:

```text
x,y,t,temperature_clean,temperature
```

`temperature` is the (possibly noisy) observation used for training; `temperature_clean` is retained for diagnostics.

---

## 8. Network architecture

### Temperature network \(T_\theta\)

Default full configuration:

```text
(x, y, t)
   → Linear(3, 64) + tanh
   → Linear(64, 64) + tanh
   → Linear(64, 64) + tanh
   → Linear(64, 64) + tanh
   → Linear(64, 1)
```

Configurable:

| Option | Examples |
| ------ | -------- |
| Hidden layers / width | `[64,64,64,64]`, `[32,32,32]`, … |
| Activation | `tanh` (default), `relu`, `gelu`, `silu` |
| Initialization | `xavier`, `kaiming` |
| Fourier features | optional random Fourier embedding of \((x,y,t)\) |

### Data-only baseline

Same MLP architecture, trained with **measurement loss only** (no PDE / IC / BC terms). This answers:

> Does physics information improve reconstruction when measurements are sparse?

---

## 9. Inverse parameter learning

Unknown diffusivity is a trainable `nn.Parameter` with a positivity transform:

\[
\alpha = \mathrm{softplus}(\alpha_{\mathrm{raw}})
= \log\bigl(1 + e^{\alpha_{\mathrm{raw}}}\bigr)
> 0
\]

Alternatively, an exponential transform \(\alpha = \exp(\alpha_{\mathrm{raw}})\) is supported.

**Important implementation detail:** optimization is performed on the **non-dimensional** diffusivity

\[
\alpha_{\mathrm{nd}} = \alpha_{\mathrm{physical}} \cdot \frac{t_f}{L_x^2}
\]

Physical \(\alpha\) is recovered after training for reporting and evaluation.

Initial guesses are configurable (e.g. \(5\times10^{-6}\), \(2\times10^{-5}\), \(5\times10^{-5}\) around a true value of \(10^{-5}\)) to study inverse robustness.

---

## 10. Loss formulation

\[
\mathcal{L}
=
\lambda_{\mathrm{pde}}\,\mathcal{L}_{\mathrm{pde}}
+
\lambda_{\mathrm{ic}}\,\mathcal{L}_{\mathrm{ic}}
+
\lambda_{\mathrm{bc}}\,\mathcal{L}_{\mathrm{bc}}
+
\lambda_{\mathrm{data}}\,\mathcal{L}_{\mathrm{data}}
+
\lambda_{\mathrm{reg}}\,\mathcal{L}_{\mathrm{reg}}
\]

| Term | Definition | Typical weight |
| ---- | ---------- | -------------: |
| \(\mathcal{L}_{\mathrm{pde}}\) | Mean squared heat residual on interior collocation points | 1 |
| \(\mathcal{L}_{\mathrm{ic}}\) | MSE vs initial temperature | 10 |
| \(\mathcal{L}_{\mathrm{bc}}\) | MSE of BC residual on boundary points | 10 |
| \(\mathcal{L}_{\mathrm{data}}\) | MSE vs sparse sensor temperatures | 50–100 |
| \(\mathcal{L}_{\mathrm{reg}}\) | Optional regularization (default off) | 0 |

All \(\lambda\) weights are YAML-configurable.

**Numerical safety:** if the total loss becomes non-finite (NaN/Inf), training raises `FloatingPointError` instead of continuing silently.

### Collocation sampling

| Point set | Purpose |
| --------- | ------- |
| Interior | PDE residual |
| Initial (\(t=0\)) | IC loss |
| Boundary edges | BC loss |
| Sensors | Data loss |

Sampling methods: **uniform**, **Latin Hypercube (LHS)**, **Sobol** (default), all seeded for reproducibility.

Optional **residual-based adaptive refinement**: after training epochs, high-residual regions receive additional collocation points.

---

## 11. Scaling & non-dimensionalization

Unscaled thermal problems with \(\alpha \sim 10^{-5}\) and \(t_f \sim 10^2\) create severe optimizer imbalance. This project normalizes:

\[
x^\star = \frac{x}{L_x},\quad
y^\star = \frac{y}{L_y},\quad
t^\star = \frac{t}{t_f},\quad
T^\star = \frac{T}{T_{\mathrm{scale}}}
\]

The trained PDE becomes:

\[
\frac{\partial T^\star}{\partial t^\star}
=
\alpha_{\mathrm{nd}}
\left(
\frac{\partial^2 T^\star}{\partial x^{\star 2}}
+ \gamma\frac{\partial^2 T^\star}{\partial y^{\star 2}}
\right)
+ q^\star
\]

where \(\gamma = (L_x/L_y)^2\) and \(\alpha_{\mathrm{nd}} = \alpha\, t_f / L_x^2\).

---

## 12. Training strategy

### Two-stage optimization

| Stage | Optimizer | Role |
| ----- | --------- | ---- |
| 1 | **Adam** | Global exploration / warm-start |
| 2 | **L-BFGS** | Local polish with strong-Wolfe line search |

### Engineering controls

- Learning-rate schedulers: `cosine`, `step`, or `none`
- Gradient clipping
- Early stopping on total loss
- Periodic checkpointing + `best.pt` / `final.pt`
- Resume via `training.resume_from_checkpoint`
- Deterministic seeds (`experiment.seed`)
- Optional MLflow logging (`experiment.use_mlflow: true`)

Tracked during training:

- component losses (PDE / IC / BC / data)
- current physical \(\alpha\)
- best total loss

---

## 13. Evaluation metrics

### Temperature field (vs FD reference)

| Metric | Formula / meaning |
| ------ | ----------------- |
| MAE | mean \(\lvert T_{\mathrm{pred}} - T_{\mathrm{true}}\rvert\) |
| RMSE | \(\sqrt{\mathrm{mean}((T_{\mathrm{pred}}-T_{\mathrm{true}})^2)}\) |
| Relative \(L^2\) | \(\lVert e\rVert_2 / \lVert T_{\mathrm{true}}\rVert_2\) |

### Parameter estimation

\[
\mathrm{relative\ error}
=
\frac{\lvert\alpha_{\mathrm{pred}}-\alpha_{\mathrm{true}}\rvert}{\alpha_{\mathrm{true}}}
\]

Also reported: absolute error and \(\alpha\) convergence history.

### Visualizations (matplotlib)

Generated under `outputs/<run>/figures/`:

- true temperature field
- PINN temperature field
- absolute error field
- sensor locations
- training losses
- \(\alpha\) convergence
- predicted vs measured temperature
- sensor time series (reference / noisy obs / PINN)
- ablation error bars

---

## 14. Baseline & ablation study

`scripts/run_ablation.py` runs:

| # | Experiment | Intent |
| - | ---------- | ------ |
| 1 | Data-only network | No physics — field fit to sensors only |
| 2 | PINN, noiseless | Full physics + clean data |
| 3 | PINN + noise | Robustness to measurement noise |
| 4 | PINN, fewer sensors | Sparse-data stress test |
| 5 | PINN, more sensors | Diminishing returns / stability |

Optional `--uncertainty` flag runs ensemble or bootstrap estimation of \(\alpha\).

---

## 15. Uncertainty estimation

Practical empirical options (not Bayesian posteriors):

| Method | Procedure |
| ------ | --------- |
| **Ensemble** | Retrain independently with different seeds |
| **Bootstrap** | Resample measurement rows, then retrain |

Example report format:

```text
alpha_estimate = 1.08e-5
alpha_std      = 0.07e-5
note: empirical uncertainty from independently trained models;
      not a calibrated Bayesian posterior
```

---

## 16. Results

> All numbers below were produced by actual local runs. They are not fabricated.

### 16.1 Primary smoke inverse run

Config: `configs/smoke.yaml`  
Artifacts: `outputs/smoke/`

| Quantity | Value |
| -------- | ----- |
| Sensors | 9 |
| Time samples / sensor | 15 |
| Noise std | 0.0 |
| True \(\alpha\) | 0.05 |
| Initial guess | 0.08 |
| Estimated \(\alpha\) | **0.04633** |
| Absolute error | 0.00367 |
| Relative parameter error | **7.34%** |
| Field MAE | 0.00972 |
| Field RMSE | 0.01414 |
| Field relative \(L^2\) | **0.0448** |
| Adam epochs | 800 + L-BFGS |

**Takeaway:** from a deliberately wrong initial guess (\(0.08\)), the inverse PINN recovered diffusivity within ~7% and reconstructed the temperature field to ~4.5% relative \(L^2\).

### 16.2 Ablation study (shorter budget)

Config: `configs/smoke_ablation.yaml` (500 Adam epochs)  
Artifacts: `outputs/smoke_ablation_ablation/`

| Experiment | Sensors | Noise | True α | Estimated α | Rel. α error | Field RMSE | Field rel. \(L^2\) |
| ---------- | ------: | ----: | -----: | ----------: | -----------: | ---------: | -----------------: |
| data_only | 9 | 0.00 | 0.05 | — | — | 0.0626 | 0.198 |
| pinn_clean | 9 | 0.00 | 0.05 | 0.0401 | 0.199 | 0.0183 | 0.058 |
| pinn_noise | 9 | 0.02 | 0.05 | 0.0402 | 0.196 | 0.0213 | 0.067 |
| pinn_sparse | 4 | 0.00 | 0.05 | 0.0467 | 0.065 | 0.0266 | 0.084 |
| pinn_dense | 15 | 0.00 | 0.05 | 0.0380 | 0.240 | 0.0183 | 0.058 |

**Observations**

1. **Physics helps field reconstruction.** PINN RMSE (~0.018–0.027) is substantially better than data-only (~0.063) under the same sparse sensors.
2. **Noise modestly degrades** field accuracy (0.018 → 0.021) without collapsing the inverse.
3. **Parameter recovery needs adequate optimization budget.** The longer smoke run reaches ~7% α error; the short ablation still shows beneficial physics for the field even when α is only partially converged.
4. **More sensors ≠ automatically better α** under a fixed short budget — optimization dynamics matter as much as data volume.

### 16.3 Full physical-\(\alpha\) benchmark

Use `configs/inverse_alpha.yaml` / `configs/noisy_inverse.yaml` (\(\alpha_{\mathrm{true}}=10^{-5}\)) for longer portfolio runs. Those configs are ready; fill additional result rows only after executing them.

---

## 17. Project structure

```text
.
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── forward.yaml              # reference / forward diagnostics
│   ├── inverse_alpha.yaml        # primary physical-α inverse benchmark
│   ├── noisy_inverse.yaml        # inverse with measurement noise
│   ├── smoke.yaml                # fast end-to-end verification
│   └── smoke_ablation.yaml       # compact ablation budget
│
├── src/inverse_pinn_thermal/
│   ├── config.py                 # Pydantic YAML schema
│   ├── geometry.py               # rectangular domain helpers
│   ├── sampling.py               # uniform / LHS / Sobol + adaptive
│   ├── cli.py                    # shared CLI entry points
│   ├── physics/
│   │   ├── heat_equation.py      # AD residual + manufactured solution
│   │   ├── boundary_conditions.py
│   │   └── initial_conditions.py
│   ├── models/
│   │   ├── pinn.py               # inverse PINN MLP
│   │   └── data_only.py          # baseline network
│   ├── inverse/
│   │   ├── parameters.py         # softplus / exp positive params
│   │   └── identifiability.py
│   ├── reference/
│   │   └── finite_difference.py  # FD solver + analytic verification
│   ├── data/
│   │   ├── sensors.py
│   │   └── noise.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py            # Adam + L-BFGS
│   │   └── checkpointing.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── ablation.py
│   │   └── uncertainty.py
│   └── visualization/
│       └── plots.py
│
├── scripts/
│   ├── generate_reference.py
│   ├── generate_measurements.py
│   ├── train_inverse.py
│   ├── train_baseline.py
│   ├── evaluate.py
│   └── run_ablation.py
│
├── tests/
│   ├── test_core.py              # physics, FD, sensors, config, ckpt
│   └── test_e2e_smoke.py         # end-to-end inverse smoke
│
└── outputs/                      # run artifacts (gitignored contents)
```

---

## 18. Installation

**Requirements:** Python **≥ 3.11**, pip.

```bash
# clone / enter the repository root
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

# Linux / macOS
# source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"

# optional experiment tracking
pip install -e ".[dev,mlflow]"
```

Core dependencies: `torch`, `numpy`, `scipy`, `matplotlib`, `pydantic`, `PyYAML`, `pandas`, `tqdm`.  
Dev dependencies: `pytest`, `ruff`, `mypy`.

---

## 19. Configuration

Experiments are fully driven by YAML. Example excerpt from `configs/inverse_alpha.yaml`:

```yaml
experiment:
  name: inverse_alpha
  seed: 42
  output_dir: outputs
  use_mlflow: false

geometry:
  length_x: 1.0
  length_y: 1.0
  final_time: 100.0

physics:
  true_alpha: 1.0e-5

inverse:
  parameter: alpha
  initial_guess: 2.0e-5
  transform: softplus

network:
  hidden_layers: [64, 64, 64, 64]
  activation: tanh

sampling:
  n_interior: 8000
  n_boundary: 2000
  n_initial: 1500
  method: sobol

sensors:
  n_sensors: 12
  n_time_samples: 40

measurement:
  noise_std: 0.0

training:
  adam_epochs: 5000
  learning_rate: 0.001
  lbfgs_enabled: true
  lbfgs_max_iter: 500

loss:
  lambda_pde: 1.0
  lambda_bc: 10.0
  lambda_ic: 10.0
  lambda_data: 100.0
```

Configs are validated with **Pydantic** on load (`load_config`).

| Config | Purpose |
| ------ | ------- |
| `configs/smoke.yaml` | Fast CI / demo end-to-end |
| `configs/smoke_ablation.yaml` | Compact ablation table |
| `configs/inverse_alpha.yaml` | Physical \(\alpha=10^{-5}\) inverse |
| `configs/noisy_inverse.yaml` | Same + measurement noise |
| `configs/forward.yaml` | Reference / forward diagnostics |

---

## 20. Exact run commands

### A. Fast end-to-end smoke (recommended first)

```bash
python scripts/generate_reference.py --config configs/smoke.yaml
python scripts/generate_measurements.py --config configs/smoke.yaml
python scripts/train_inverse.py --config configs/smoke.yaml
python scripts/train_baseline.py --config configs/smoke.yaml
python scripts/evaluate.py --run outputs/smoke
```

### B. Primary physical-\(\alpha\) inverse benchmark

```bash
python scripts/generate_reference.py --config configs/inverse_alpha.yaml
python scripts/generate_measurements.py --config configs/inverse_alpha.yaml
python scripts/train_inverse.py --config configs/inverse_alpha.yaml
python scripts/evaluate.py --run outputs/inverse_alpha
```

### C. Noisy inverse

```bash
python scripts/generate_reference.py --config configs/noisy_inverse.yaml
python scripts/generate_measurements.py --config configs/noisy_inverse.yaml
python scripts/train_inverse.py --config configs/noisy_inverse.yaml
python scripts/evaluate.py --run outputs/noisy_inverse
```

### D. Ablation (+ optional uncertainty)

```bash
python scripts/run_ablation.py --config configs/smoke_ablation.yaml
python scripts/run_ablation.py --config configs/smoke_ablation.yaml --uncertainty
```

### E. Quality checks

```bash
ruff check .
pytest
pytest -m slow          # includes E2E training test
mypy src/inverse_pinn_thermal
```

After editable install, console scripts are also available:

```bash
ipt-generate-reference --config configs/smoke.yaml
ipt-generate-measurements --config configs/smoke.yaml
ipt-train-inverse --config configs/smoke.yaml
ipt-train-baseline --config configs/smoke.yaml
ipt-evaluate --run outputs/smoke
ipt-run-ablation --config configs/smoke_ablation.yaml
```

---

## 21. Outputs & artifacts

Each run directory (e.g. `outputs/smoke/`) typically contains:

| Artifact | Contents |
| -------- | -------- |
| `reference.pt` | FD grid + temperature tensor |
| `measurements.csv` | Sparse sensor observations |
| `config_resolved.yaml` | Full validated config snapshot |
| `history.csv` | Loss components + α trajectory |
| `summary.json` | True/estimated α and errors |
| `evaluation.json` | Field + parameter metrics |
| `best.pt` / `final.pt` | Checkpoints (weights, α, seed, history) |
| `figures/` | PNG visualizations |

Checkpoint payload includes: network weights, learned thermal parameters, optimizer state (when saved), epoch, seed, config, and loss history — enabling resume and auditability.

---

## 22. Testing & quality

| Suite | What it verifies |
| ----- | ---------------- |
| Manufactured-solution test | Heat residual ≈ 0 for analytic \(T\) |
| Autograd derivative test | First/second derivatives match analytic modal field |
| Boundary modules | Dirichlet / Neumann / Robin residual shapes |
| Parameter positivity | softplus / exp transforms stay \(>0\) |
| Sensor + noise | Placement bounds and Gaussian noise |
| Sampling | uniform / LHS / Sobol shapes |
| FD verification | Solver vs analytic modal solution |
| CFL guard | Unstable grids raise clearly |
| Config validation | Invalid dimension rejected by Pydantic |
| Checkpoint round-trip | Save/load network state |
| E2E smoke (`-m slow`) | Reference → sensors → train → α recovery |

Linting / typing:

```bash
ruff check .
mypy src/inverse_pinn_thermal
```

---

## 23. Reproducibility

- Fixed `experiment.seed` for NumPy / PyTorch
- Seeded Sobol / LHS / uniform collocation
- Seeded sensor placement and noise draws
- Resolved config written beside every run
- CFL-enforced reference solver (no silent unstable grids)
- Explicit failure on NaN losses
- Results tables in this README filled only from executed runs

---

## 24. Limitations

1. **FD reference BC scope:** explicit solver currently supports homogeneous Dirichlet for generation/validation.
2. **Single-parameter focus:** joint \((\alpha,k,h)\) estimation is intentionally deferred due to identifiability.
3. **Empirical uncertainty only:** ensemble/bootstrap spreads are not calibrated Bayesian posteriors.
4. **Compute vs physics scale:** physical \(\alpha\sim10^{-5}\) cases need larger collocation budgets and longer Adam/L-BFGS than the smoke demo.
5. **Adaptive sampling:** residual refinement is implemented but optional/lightweight in Version 1.
6. **Short-budget ablations** can under-converge \(\alpha\) even when field RMSE already improves — report training budget with every table.

---

## 25. Next improvements

1. Run full `inverse_alpha.yaml` / `noisy_inverse.yaml` trainings for portfolio-grade physical-\(\alpha\) tables.
2. Add Crank–Nicolson / implicit FD and Neumann–Robin reference cases.
3. Initialization-robustness sweep over multiple \(\alpha\) guesses with a summary figure.
4. Turn on residual adaptive sampling by default with a clear refine schedule.
5. Enable MLflow tracking for multi-seed campaign dashboards.
6. Optional Bayesian / Laplace approximation for calibrated parameter uncertainty.
7. 3D or anisotropic diffusivity extensions once 2D inverse reliability is established.

---

## 26. License

MIT

---

### Portfolio blurb

> I designed an inverse Physics-Informed Neural Network project to estimate thermal diffusivity from sparse, noisy temperature sensors while enforcing the 2D transient heat equation. I defined the scientific problem, validation strategy, and experiments; the code was implemented with the help of an AI coding assistant under my direction. The repository includes a validated finite-difference reference solver, a data-only baseline, ablation studies, empirical uncertainty estimation, and a reproducible Python package with tests.
