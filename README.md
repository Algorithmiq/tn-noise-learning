# tn-noise-learning

[![Tests](https://github.com/Algorithmiq/tn-noise-learning/actions/workflows/test.yml/badge.svg)](https://github.com/Algorithmiq/tn-noise-learning/actions/workflows/test.yml)
[![Documentation](https://github.com/Algorithmiq/tn-noise-learning/actions/workflows/docs.yml/badge.svg)](https://algorithmiq.github.io/tn-noise-learning/)
[![codecov](https://codecov.io/gh/Algorithmiq/tn-noise-learning/graph/badge.svg)](https://codecov.io/gh/Algorithmiq/tn-noise-learning)
[![PyPI](https://img.shields.io/pypi/v/tn-noise-learning)](https://pypi.org/project/tn-noise-learning/)

Learn the noise channel of a quantum circuit layer from measurement shots, with tensor networks.

The channel is represented as a **locally purified density operator (LPDO)**: one Kraus tensor per qubit, joined
by virtual bonds along a chosen **topology** (a 1D chain, a 2D square lattice, or any custom graph). The LPDO is
completely positive by construction and is trained by maximum likelihood with JAX and optax, with a soft
trace-preservation penalty. Datasets are plain [zarr](https://zarr.dev) groups.

> [!WARNING]
> This package is under active development and follows [Semantic Versioning](https://semver.org/). While in `0.x.y`,
> breaking changes may occur in minor releases. Pin your version if you depend on it.

📖 **Documentation:** <https://algorithmiq.github.io/tn-noise-learning/>

## Installation

```bash
pip install tn-noise-learning           # or: uv add tn-noise-learning
pip install "tn-noise-learning[plot]"   # with plotting
```

Python 3.12 to 3.15.

## Usage

```python
from qiskit import QuantumCircuit
from tn_noise_learning import (
    LPDO,
    NoiseDataset,
    Topology,
    Trainer,
    TrainingConfig,
    sample_shots,
    two_qubit_depolarizing,
)

topology = Topology.square(2, 2)  # or Topology.chain(n), Topology.from_edges([...])
circuit = QuantumCircuit(4)
circuit.cz(0, 1)
circuit.cz(2, 3)

truth = two_qubit_depolarizing(topology, p=0.05, edges=[(0, 1), (2, 3)])
sample_shots(truth, circuit, num_settings=300, shots_per_setting=100, seed=0).save("example.zarr")

data = NoiseDataset.load("example.zarr")  # your own data goes in this format
train, test = data.split(test_fraction=0.1, seed=0)

model = LPDO.random(topology, bond_dim=4, kraus_dim=4, seed=0)
history = Trainer(TrainingConfig(epochs=30, batch_size=250)).fit(model, train, test)

history.plot("history.png")  # loss, NLL, TP penalty, trace, distance to ideal (if known)
model.save("model.zarr")
```

`examples/minimal_example.py` is this script; it runs in under a minute.

## What is in the box

| Module | Purpose |
| --- | --- |
| `topology` | `Topology`: sites and bond edges; `chain`, `square`, `from_edges` |
| `lpdo` | `LPDO`: Kraus tensors on a topology; random / identity init, trace, distance, dense form, zarr I/O |
| `circuits` | Ideal circuit as a tensor network on any connectivity (operator-Schmidt gate decomposition) |
| `dataset` | `NoiseDataset`: shot labels + single-qubit states/effects in zarr; `split`, `batches` |
| `channels` | Noise channels as LPDOs: `depolarizing`, `pauli_channel`, `two_qubit_depolarizing`; Pauli eigenstates |
| `simulate` | `sample_shots`: exact per-setting outcome distributions of a known channel |
| `losses` | Transition probabilities, negative log-likelihood, trace-preservation penalty |
| `trainer` | `TrainingConfig`, `Trainer`: jitted optax loop, L-BFGS-B warm start, early stopping, model selection |
| `history`, `plotting` | `History` with zarr I/O and `plot_history` |

## Development

The repository ships a devcontainer and a Nix dev shell; with neither, install [uv](https://docs.astral.sh/uv/)
and [just](https://github.com/casey/just):

```bash
just build --all-groups && uv run prek install
just test
just lint
```

See the [contributing guide](https://algorithmiq.github.io/tn-noise-learning/contributing/).

## Contributing

Read [CONTRIBUTING.md](https://github.com/Algorithmiq/tn-noise-learning/blob/main/CONTRIBUTING.md) before opening a
pull request. Opening a pull request confirms that you agree to the
[Contributor License Agreement](https://github.com/Algorithmiq/tn-noise-learning/blob/main/CLA.md).

## Citation

If you use `tn-noise-learning` in your research, please cite it using
[CITATION.cff](https://github.com/Algorithmiq/tn-noise-learning/blob/main/CITATION.cff) (GitHub's "Cite this
repository" button), and the papers in the [references](https://algorithmiq.github.io/tn-noise-learning/references/).

## License

`tn-noise-learning` is released under the
[Apache License 2.0](https://github.com/Algorithmiq/tn-noise-learning/blob/main/LICENSE).
