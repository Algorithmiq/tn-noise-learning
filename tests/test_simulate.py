from __future__ import annotations

from functools import reduce
from itertools import product

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from tn_noise_learning.channels import PAULI_EIGENSTATES, depolarizing, pauli_eigenstates, two_qubit_depolarizing
from tn_noise_learning.simulate import outcome_distribution, sample_shots
from tn_noise_learning.topology import Topology


def test_outcome_distribution_matches_dense():
    topology = Topology.chain(3)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rzz(0.4, 1, 2)
    channel = two_qubit_depolarizing(topology, 0.3, edges=[(0, 1)])
    states, povms = pauli_eigenstates(3), pauli_eigenstates(3)
    prep, basis = [2, 0, 5], [1, 2, 0]
    probs = outcome_distribution(channel, qc, prep, basis, states, povms)
    assert probs.shape == (2, 2, 2)
    assert probs.sum() == pytest.approx(1.0)

    u = Operator(qc.reverse_bits()).data
    rho = reduce(np.kron, [states[j, prep[j]] for j in range(3)])
    out = (channel.to_dense() @ (u @ rho @ u.conj().T).reshape(-1)).reshape(8, 8)
    for bits in product((0, 1), repeat=3):
        effect = reduce(np.kron, [povms[j, 2 * basis[j] + bits[j]] for j in range(3)])
        assert probs[bits] == pytest.approx(np.real(np.trace(effect @ out)), abs=1e-12)


def test_sample_shots_reproducible_and_valid():
    channel = depolarizing(Topology.square(2, 2), 0.1)
    data = sample_shots(channel, num_settings=7, shots_per_setting=5, seed=3)
    again = sample_shots(channel, num_settings=7, shots_per_setting=5, seed=3)
    assert len(data) == 35
    assert data.ideal_channel is channel
    np.testing.assert_array_equal(data.inputs, again.inputs)
    np.testing.assert_array_equal(data.outcomes, again.outcomes)
    assert data.inputs.min() >= 0 and data.inputs.max() < 6
    assert data.outcomes.min() >= 0 and data.outcomes.max() < 6


def test_sample_shots_frequencies_follow_distribution():
    channel = depolarizing(Topology.chain(1), 0.2)
    prep, basis = [2], [0]
    data = sample_shots(channel, settings=[(prep, basis)], shots_per_setting=20000, seed=0)
    expected = outcome_distribution(channel, None, prep, basis, pauli_eigenstates(1), pauli_eigenstates(1))
    frequency = np.mean(data.outcomes[:, 0] == 1)
    assert frequency == pytest.approx(expected[1], abs=0.02)


def test_sample_shots_rejects_bad_povms():
    povms = np.stack([PAULI_EIGENSTATES[[0, 2, 4]]])
    with pytest.raises(ValueError, match="sum to the identity"):
        sample_shots(depolarizing(Topology.chain(1), 0.1), povms=povms, num_settings=1)
