# tn-noise-learning

Tensor-network learning of quantum noise channels (LPDO + MPO + JAX) for
quantum process tomography.

This is the engineered/reusable extract of the research codebase **SABRINA**
(*Sensible Approach to Beat Ridiculously Impractical Noise-characterization
Algorithms*).

## Install

```bash
uv sync                  # runtime + tests
uv sync --all-extras     # also include plotting helpers
```

Python 3.12 is required. Pinned to the package versions sabrina was last
validated against (numpy 2.3, qiskit 2.2, quimb 1.11, jax 0.9, optax 0.2.8,
cotengra 0.7).

## What's here

```
src/tn_noise_learning/
├── tn_core/        # LPDO super-operator class + MPO/LPDO utilities
├── noise_models/   # Noise channel constructors (LPDO and MPO)
├── qpt/            # QPT simulation pipeline + SPAM variant + data generation CLI
├── training/       # JAX/optax optimisation of LPDO parameters
├── io/             # JSON serialisation for experiments and results
└── utils/          # PTM conversions and channel metrics
```

`examples/` holds the original research notebooks and per-figure analysis
scripts, ported verbatim — see `examples/README.md`.

## Quick start

```python
import qiskit as qk
from tn_noise_learning import (
    select_noise_channel, build_qpt_mpo, simulate_experiments, train,
)

# 1. Build the QPT MPO for an ideal circuit + a known noise model.
qc = qk.QuantumCircuit(4)
qc.cx(0, 1); qc.cx(2, 3)
noise_mpo = select_noise_channel("even_depol1e-3_odd_depol5e-4", num_qubits=4, tensor_type="mpo")
qpt_mpo, states, povms = build_qpt_mpo(qc_unitary=qc, noise_mpo=noise_mpo)

# 2. Sample synthetic measurement shots.
experiments = simulate_experiments(qpt_mpo, num_shots=100)

# 3. (or end-to-end) train an LPDO on a saved dataset:
#    metrics = train("path/to/experiments.json")
```

CLI for synthetic data generation:

```bash
uv run python -m tn_noise_learning.qpt.data_generation \
    --num-qubits 6 --noise even_depol1e-3_odd_depol5e-4 \
    --num-shots 5000 --strategy global_repeated
```

## Develop

```bash
uv run pytest -q                # smoke tests
uv run ruff check .             # lint
uv run ruff format --check .    # formatting
uv run mkdocs serve             # local docs preview
```

## Documentation

Conceptual overview and full auto-generated API reference live in `docs/`.
Build locally with `uv run mkdocs build`.
