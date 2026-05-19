# Examples

This directory holds research scripts and notebooks ported from the original
`sabrina/` repository. They are **not** installed as part of
`tn_noise_learning`. They serve two purposes:

1. **Demonstration** — show how to use the library on realistic problems.
2. **Reproducibility material** — the original analyses behind the paper plots.

## Layout

| Folder | What it does |
| --- | --- |
| `notebooks/` | The three top-level notebooks from sabrina, copied verbatim: `noise_learning.ipynb` (training pipeline), `generate_qpt_data.ipynb` (synthetic data generation), `noise_tests.ipynb` (validation tests). |
| `data/` | Sparse Pauli-Lindblad coefficient files used by several examples and notebooks. |
| `accuracy_vs_coefficients/` | Scaling study: fidelity vs. number of Pauli-Lindblad coefficients. |
| `accuracy_vs_qubits/` | Scaling study: fidelity vs. number of qubits. |
| `accuracy_vs_shots/` | Scaling study: fidelity vs. number of measurement shots. |
| `chemistry_circuits/` | LPDO learning on quantum chemistry circuits. |
| `floquet/` | Periodic (Floquet) drive noise channel experiments. |
| `global_vs_local/` | Comparison of global vs. local measurement sampling strategies. |
| `middle_out/` | Middle-out tensor contraction analysis. |
| `plot_spl_coeffs/` | Plotting helpers for sparse Pauli-Lindblad coefficients. |
| `spam/` | State-preparation and measurement (SPAM) error analyses. |
| `strong_depolarizing/` | Learning under strong depolarising noise. |
| `training_runs/` | Plots of training trajectories across multiple runs. |

## Running an example

These scripts and notebooks were originally written against bare-module
imports (e.g. `import lpdo_utils`). When run inside this repo they need to
import from the installed package instead, e.g.:

```python
from tn_noise_learning.tn_core import build_lpdo, lpdo_to_mpo
from tn_noise_learning.noise_models import select_noise_channel
from tn_noise_learning.training import train
```

Many scripts also reference paths like `data/simulated_experiments/...` and
`opt_results/...` that lived inside the sabrina working directory. Those
directories are gitignored in the new repo and need to be regenerated locally
by running `examples/notebooks/generate_qpt_data.ipynb` first.

> **Note on `floquet/noise_learning.py`**: this is a stale, divergent copy of
> the training script and is not maintained. It is preserved verbatim for
> reproducibility of the Floquet analysis only — prefer
> [`tn_noise_learning.training.train`][tn_noise_learning.training.train] for new work.
