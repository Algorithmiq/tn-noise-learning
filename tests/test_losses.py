from __future__ import annotations

from functools import reduce
from itertools import product

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from tn_noise_learning.channels import PAULI_EIGENSTATES, two_qubit_depolarizing
from tn_noise_learning.circuits import circuit_network
from tn_noise_learning.lpdo import LPDO
from tn_noise_learning.losses import negative_log_likelihood, probabilities, tp_penalty
from tn_noise_learning.topology import Topology


def random_labels(rng, num_sites, batch):
    inputs = rng.integers(0, 6, size=(batch, num_sites))
    bases = rng.integers(0, 3, size=(batch, num_sites))
    return inputs, bases


def gather(table, labels):
    return np.stack([table[labels[:, j]] for j in range(labels.shape[1])])


def test_identity_channel_outcomes_sum_to_one():
    rng = np.random.default_rng(0)
    topology = Topology.square(2, 2)
    inputs, bases = random_labels(rng, 4, 5)
    states = gather(PAULI_EIGENSTATES, inputs)
    lpdo_tn = LPDO.identity(topology).network()
    total = sum(
        probabilities(lpdo_tn, states, gather(PAULI_EIGENSTATES, 2 * bases + np.array(bits)))
        for bits in product((0, 1), repeat=4)
    )
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


def test_probabilities_match_dense_with_circuit():
    rng = np.random.default_rng(1)
    topology = Topology.chain(3)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)
    channel = two_qubit_depolarizing(topology, 0.2, edges=[(1, 2)])
    inputs, bases = random_labels(rng, 3, 4)
    states = gather(PAULI_EIGENSTATES, inputs)
    povms = gather(PAULI_EIGENSTATES, 2 * bases + rng.integers(0, 2, size=bases.shape))
    probs = probabilities(channel.network(), states, povms, circuit_network(qc))

    superop = channel.to_dense()
    u = Operator(qc.reverse_bits()).data
    for shot in range(4):
        rho = reduce(np.kron, states[:, shot])
        rho = u @ rho @ u.conj().T
        out = (superop @ rho.reshape(-1)).reshape(8, 8)
        expected = np.real(np.trace(reduce(np.kron, povms[:, shot]) @ out))
        assert probs[shot] == pytest.approx(expected, abs=1e-12)


def test_negative_log_likelihood():
    probs = np.array([0.5, 0.25])
    assert negative_log_likelihood(None, probs) == pytest.approx(-(np.log(0.5) + np.log(0.25)) / 2)


def test_tp_penalty_zero_for_identity_and_positive_otherwise():
    topology = Topology.square(2, 2)
    assert tp_penalty(LPDO.identity(topology).network()) == pytest.approx(0.0, abs=1e-12)
    assert tp_penalty(two_qubit_depolarizing(topology, 0.1, edges=[(0, 1), (2, 3)]).network()) == pytest.approx(0.0, abs=1e-12)
    assert tp_penalty(LPDO.random(topology, 2, 2, seed=0).network()) > 1e-3
