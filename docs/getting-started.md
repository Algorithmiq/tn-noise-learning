# Getting started

## Install

```bash
uv sync                  # install runtime + test deps
uv sync --all-extras     # also pull plotting helpers
```

Python 3.12 is required. The library targets the package versions listed in
sabrina's `CLAUDE.md` (numpy 2.3, qiskit 2.2, quimb 1.11, jax 0.9, optax 0.2.8,
cotengra 0.7); other versions may produce silent breakage.

## A 4-qubit toy run

```python
import qiskit as qk
from tn_noise_learning import (
    select_noise_channel, build_qpt_mpo, simulate_experiments,
)

num_qubits = 4
qc = qk.QuantumCircuit(num_qubits)
qc.cx(0, 1)
qc.cx(2, 3)

noise_mpo = select_noise_channel(
    "even_depol1e-3_odd_depol5e-4",
    num_qubits=num_qubits,
    tensor_type="mpo",
)

qpt_mpo, states, povms = build_qpt_mpo(qc_unitary=qc, noise_mpo=noise_mpo)
experiments = simulate_experiments(qpt_mpo, num_shots=100)
```

## Generate a dataset from the CLI

```bash
uv run python -m tn_noise_learning.qpt.data_generation \
    --num-qubits 6 \
    --noise even_depol1e-3_odd_depol5e-4 \
    --num-shots 5000 \
    --strategy global_repeated \
    --output-dir data/simulated_experiments
```

## Full training loop

```python
from tn_noise_learning import train

metrics = train(
    data_file="cxeven_nq6_even_depol1e-3_odd_depol5e-4_ns5000_global_repeated_sic_states_2026-05-19.json",
    virtual_bd=4,
    kraus_bd=4,
    batch_size=250,
    epochs=20,
)
```

See `examples/notebooks/` for the original end-to-end notebooks.
