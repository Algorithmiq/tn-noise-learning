# CLAUDE.md

Guidance for Claude Code (and other LLM tooling) working in this repository.

## Project

`tn_noise_learning` — tensor-network learning of quantum noise channels.

Quantum Process Tomography (QPT) implemented with tensor networks: noise
channels are represented as MPOs or LPDOs in Pauli Transfer Matrix (PTM)
form, then optimised with JAX autodiff. Ported from the research codebase
**SABRINA**.

## Environment

Python 3.12, dependencies managed by `uv`.

```bash
uv sync                  # install runtime + test deps
uv run pytest -q
uv run ruff check .
uv run mkdocs build
```

The pinned versions are taken from sabrina's "known working" set:
numpy 2.3, qiskit 2.2, quimb 1.11, jax 0.9, optax 0.2.8, cotengra 0.7.

## Architecture

```
src/tn_noise_learning/
├── tn_core/        # LPDOSuperOperator1D + LPDO/MPO construction helpers
├── noise_models/   # select_noise_channel dispatcher + LPDO/MPO channel constructors
├── qpt/            # build_qpt_mpo, simulate_experiments, SPAM variant, generate_data CLI
├── training/       # JAX loss + optimize_tn + dataset pipeline + train() driver
├── io/             # save_results / load_results (complex→string serialisation)
└── utils/          # PTM conversions and channel metrics
```

End-to-end flow:

```
input states ─┐
unitary MPO   ├─▶ build_qpt_mpo()  ─▶  simulate_experiments()  ─▶  JSON data
noise MPO/LPDO┘                                                       │
                                                          create_training_dataset()
                                                                      │
                                                              optimize_tn()
                                                                      │
                                                          trained LPDO + metrics
                                                                      │
                                                              save_results()
```

## Key concepts

* **PTM** (Pauli Transfer Matrix) — density matrices encoded as `4^n`-element
  vectors; channels become super-operators on 4-D physical legs.
* **MPO** — Matrix Product Operator; bond dimension χ controls expressivity.
* **LPDO** — Locally Purified Density Operator; Kraus dimension K controls
  Kraus rank. Complete positivity by construction.
* **Trace preservation (TP)** — enforced as a soft penalty
  `λ · ‖Tr[E] − 𝟙‖ / 2^{n/2}` and optionally warm-started by L-BFGS-B via
  `optimize_trace_preservation`.
* **cotengra cache** (`ctg_cache/`) — persistent contraction-path cache used
  by training; gitignored, safe to delete.

## Working in this repo

* `src/` is the library; `tests/` mirrors its tree.
* `examples/` is research scripts/notebooks ported verbatim from sabrina —
  **not** installed; use it only for reproducibility material.
* No `data/` or `opt_results/` directories are committed — they are
  regenerated locally by running `examples/notebooks/generate_qpt_data.ipynb`
  or the data-generation CLI.

### Style

* Google-style docstrings, full type hints encouraged but not enforced yet
  (`mypy --strict` is a follow-up milestone).
* Ruff configured at 120 cols, `from __future__ import annotations` required.
* Quantum identifiers like `N`, `K`, `U` are permitted as variable names
  (uppercase) — see `[tool.ruff.lint] ignore = ["N803", "N806"]`.

### Notes carried over from sabrina

The original codebase targeted older package versions. The migration was a
verbatim move of the logic with minimal cleanup; known patterns:

* `qiskit 2.x` API changes — most call sites already use the new paths but
  watch for `_data` / `_index` private accesses in `mpo_utils.qc_to_mpo`.
* `numpy 2.x` — broadcasts and type aliases.
* `jax 0.9.x` — `jax.config.update("jax_enable_x64", True)` is called in
  `training/losses.py` and `training/pipeline.py`.

### Stale code flagged for follow-up

* `examples/floquet/noise_learning.py` is a stale, divergent copy of the
  training script kept only for reproducibility of the Floquet analysis.
  Prefer `tn_noise_learning.training.train` for new work.
