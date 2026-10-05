from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from tn_noise_learning.channels import depolarizing, two_qubit_depolarizing
from tn_noise_learning.lpdo import LPDO
from tn_noise_learning.simulate import sample_shots
from tn_noise_learning.topology import Topology
from tn_noise_learning.trainer import Trainer, TrainingConfig


def recover(truth, circuit, bond_dim, kraus_dim, epochs):
    """Statistical floor with 30k shots is about 1e-3 (chain) and 2e-3 (square) in squared normalised distance."""
    data = sample_shots(truth, circuit, num_settings=300, shots_per_setting=100, seed=0)
    train, test = data.split(0.1, seed=0)
    model = LPDO.random(truth.topology, bond_dim=bond_dim, kraus_dim=kraus_dim, seed=0)
    history = Trainer(TrainingConfig(epochs=epochs, batch_size=250, tp_warmstart_steps=200, patience=epochs)).fit(model, train, test)
    return model, history


def test_recover_depolarizing_chain():
    truth = depolarizing(Topology.chain(2), 0.1)
    model, history = recover(truth, None, bond_dim=2, kraus_dim=4, epochs=30)
    assert model.distance(truth) < 3e-3
    assert abs(model.trace() - 4) < 0.08
    assert history.metrics["distance_to_ideal"][history.best_epoch - 1] == pytest.approx(model.distance(truth), rel=1e-6)


@pytest.mark.slow
def test_recover_two_qubit_depolarizing_square():
    circuit = QuantumCircuit(4)
    circuit.cz(0, 1)
    circuit.cz(2, 3)
    truth = two_qubit_depolarizing(Topology.square(2, 2), 0.08, edges=[(0, 1), (2, 3)])
    model, _ = recover(truth, circuit, bond_dim=4, kraus_dim=4, epochs=40)
    assert model.distance(truth) < 5e-3
    assert abs(model.trace() - 16) < 0.32
