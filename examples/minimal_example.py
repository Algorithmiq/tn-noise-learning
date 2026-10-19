# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Learn a two-qubit depolarizing channel on a 2x2 lattice from simulated shots.

Run with ``uv run python examples/minimal_example.py``; writes ``example.zarr``, ``model.zarr`` and ``history.png``.
"""

from __future__ import annotations

import logging

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


def main() -> None:
    """Simulate, train and plot."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Topology and the ideal circuit whose noise we want to learn.
    topology = Topology.square(2, 2)
    circuit = QuantumCircuit(4)
    circuit.cz(0, 1)
    circuit.cz(2, 3)

    # 2. Data: here simulated from a known channel and written in the zarr format the library reads.
    truth = two_qubit_depolarizing(topology, p=0.05, edges=[(0, 1), (2, 3)])
    sample_shots(truth, circuit, num_settings=300, shots_per_setting=100, seed=0).save("example.zarr")
    data = NoiseDataset.load("example.zarr")
    train, test = data.split(test_fraction=0.1, seed=0)

    # 3. Model and training.
    model = LPDO.random(topology, bond_dim=4, kraus_dim=4, seed=0)
    history = Trainer(TrainingConfig(epochs=30, batch_size=250)).fit(model, train, test)

    # 4. Results: the distance to the ideal channel is reported because the dataset carries it.
    history.plot("history.png")
    model.save("model.zarr")
    distance = min(history.metrics["distance_to_ideal"])
    print(f"best epoch {history.best_epoch}: distance to ideal {distance:.2e}, trace {model.trace():.3f}")


if __name__ == "__main__":
    main()
