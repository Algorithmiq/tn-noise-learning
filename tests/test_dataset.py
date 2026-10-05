from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from tn_noise_learning.channels import depolarizing, pauli_eigenstates
from tn_noise_learning.dataset import NoiseDataset
from tn_noise_learning.topology import Topology


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    qc = QuantumCircuit(4)
    qc.cz(0, 1)
    qc.rzz(0.5, 0, 3)
    qc.sx(1)
    return NoiseDataset(
        inputs=rng.integers(0, 6, size=(23, 4)),
        outcomes=rng.integers(0, 6, size=(23, 4)),
        states=pauli_eigenstates(4),
        povms=pauli_eigenstates(4),
        circuit=qc,
        ideal_channel=depolarizing(Topology.square(2, 2), 0.1),
    )


def test_zarr_round_trip(tmp_path, dataset):
    dataset.save(tmp_path / "data.zarr")
    loaded = NoiseDataset.load(tmp_path / "data.zarr")
    np.testing.assert_array_equal(loaded.inputs, dataset.inputs)
    np.testing.assert_array_equal(loaded.outcomes, dataset.outcomes)
    np.testing.assert_array_equal(loaded.states, dataset.states)
    assert loaded.states.dtype == np.complex128
    assert Operator(loaded.circuit).equiv(Operator(dataset.circuit))
    assert loaded.ideal_channel.distance(dataset.ideal_channel) == pytest.approx(0.0, abs=1e-12)


def test_round_trip_without_optional_fields(tmp_path, dataset):
    dataset.circuit = None
    dataset.ideal_channel = None
    dataset.save(tmp_path / "data.zarr")
    loaded = NoiseDataset.load(tmp_path / "data.zarr")
    assert loaded.circuit is None
    assert loaded.ideal_channel is None


def test_split_is_disjoint_cover(dataset):
    train, test = dataset.split(test_fraction=0.2, seed=1)
    assert len(train) + len(test) == len(dataset)
    assert len(test) == 5
    assert train.circuit is dataset.circuit


def test_batches_shapes_and_determinism(dataset):
    first = list(dataset.batches(batch_size=5, seed=3))
    second = list(dataset.batches(batch_size=5, seed=3))
    assert len(first) == 4
    assert first[0].states.shape == (4, 5, 2, 2)
    np.testing.assert_array_equal(first[1].povms, second[1].povms)
    assert len(list(dataset.batches(batch_size=5, drop_last=False))) == 5


def test_gather_matches_labels(dataset):
    batch = dataset.gather(np.array([0, 1]))
    np.testing.assert_array_equal(batch.states[2, 1], dataset.states[2, dataset.inputs[1, 2]])
    np.testing.assert_array_equal(batch.povms[3, 0], dataset.povms[3, dataset.outcomes[0, 3]])


def test_validation():
    with pytest.raises(ValueError, match="outside the range"):
        NoiseDataset(np.array([[7]]), np.array([[0]]), pauli_eigenstates(1), pauli_eigenstates(1))
    with pytest.raises(ValueError, match="Circuit acts on"):
        NoiseDataset(
            np.array([[0]]), np.array([[0]]), pauli_eigenstates(1), pauli_eigenstates(1), circuit=QuantumCircuit(2)
        )
