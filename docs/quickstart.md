# Quickstart

```bash
uv sync --all-extras
uv run python examples/minimal_example.py
```

The script below is `examples/minimal_example.py`.

## 1. Topology and circuit

```python
from qiskit import QuantumCircuit
from tn_noise_learning import Topology

topology = Topology.square(2, 2)      # sites 0..3, row-major; edges (0,1) (0,2) (1,3) (2,3)
circuit = QuantumCircuit(4)
circuit.cz(0, 1)
circuit.cz(2, 3)
```

`Topology.chain(n)` and `Topology.from_edges([...])` are the other constructors. Site integers are the qubit
order of the dataset and of the Qiskit circuit. The circuit is the *ideal* operation; the noise that follows it
is what gets learned. Gates may act on any pair of qubits, not only on topology edges.

## 2. Data

```{.python continuation}
from tn_noise_learning import NoiseDataset, sample_shots, two_qubit_depolarizing

truth = two_qubit_depolarizing(topology, p=0.05, edges=[(0, 1), (2, 3)])
sample_shots(truth, circuit, num_settings=300, shots_per_setting=100, seed=0).save("example.zarr")

data = NoiseDataset.load("example.zarr")
train, test = data.split(test_fraction=0.1, seed=0)
```

A `NoiseDataset` holds, per shot and per qubit, the label of the prepared state and of the measured effect, plus
the single-qubit `states[j]` and `povms[j]` those labels refer to, the ideal circuit (OpenQASM 2), and optionally
the ideal channel. Build one from your own experiment by filling those arrays; see
[`NoiseDataset`][tn_noise_learning.dataset.NoiseDataset] for the zarr layout.

## 3. Model and training

```{.python continuation}
from tn_noise_learning import LPDO, Trainer, TrainingConfig

model = LPDO.random(topology, bond_dim=4, kraus_dim=4, seed=0)
history = Trainer(TrainingConfig(epochs=30, batch_size=250)).fit(model, train, test)
```

`fit` warm-starts the random model onto the trace-preserving manifold with L-BFGS-B, then runs Adam on
`NLL + tp_weight · TP²`. It records per-epoch metrics, stops early when the test loss stops improving, and writes
the best epoch's tensors back into `model`. Every knob lives in
[`TrainingConfig`][tn_noise_learning.trainer.TrainingConfig]; for example

```{.python continuation}
import optax

TrainingConfig(
    learning_rate=optax.cosine_decay_schedule(1e-2, decay_steps=5000),
    tp_weight=lambda step: 1000.0 if step < 500 else 100.0,
    metrics={"bond_dim": lambda m: m.bond_dim},
)
```

## 4. Results

```{.python continuation}
history.plot("history.png")
model.save("model.zarr")
```

The figure shows train/test loss, NLL, TP penalty, trace (with the `2**n` reference) and, because the dataset
carries the ideal channel, the squared normalised distance to it. `LPDO.load("model.zarr")` restores the model.
