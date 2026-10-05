"""Temporary checks of the new topology-generic code against the legacy 1D implementation."""

from __future__ import annotations

import numpy as np
import pytest
import quimb.tensor as qtn
from qiskit import QuantumCircuit

from tn_noise_learning.channels import PAULI_EIGENSTATES, two_qubit_depolarizing
from tn_noise_learning.circuits import circuit_network
from tn_noise_learning.lpdo import LPDO
from tn_noise_learning.losses import negative_log_likelihood, probabilities, tp_penalty
from tn_noise_learning.noise_models.lpdo_channels import dep2loc_even
from tn_noise_learning.tn_core.lpdo_utils import build_lpdo, lpdo_fidelity, trace_lpdo
from tn_noise_learning.topology import Topology
from tn_noise_learning.training.dataset import evolve_hyper_tn, experiments_to_hyper_tn
from tn_noise_learning.training.losses import loss_fn


@pytest.fixture
def chain_arrays():
    rng = np.random.default_rng(0)
    shapes = [(2, 3, 2, 2), (2, 2, 3, 2, 2), (2, 3, 2, 2)]
    return [rng.standard_normal(s) + 1j * rng.standard_normal(s) for s in shapes]


def test_lpdo_matches_legacy_chain(chain_arrays):
    new = LPDO(Topology.chain(3), chain_arrays)
    old = build_lpdo(tn_data=chain_arrays)
    assert new.trace() == pytest.approx(float(np.real(trace_lpdo(old))))
    rows = [f"kn{j}" for j in range(3)] + [f"bn{j}" for j in range(3)]
    cols = [f"k{j}" for j in range(3)] + [f"b{j}" for j in range(3)]
    np.testing.assert_allclose(new.to_dense(), old.contract().to_dense(rows, cols), rtol=1e-10, atol=1e-10)
    other = LPDO.random(Topology.chain(3), bond_dim=2, kraus_dim=3, seed=5)
    old_other = build_lpdo(tn_data=other.tensors)
    assert new.distance(other) == pytest.approx(float(lpdo_fidelity(old, old_other)), rel=1e-8)


def test_loss_matches_legacy_loss_fn(chain_arrays):
    rng = np.random.default_rng(1)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)
    inputs = rng.integers(0, 6, size=(5, 3))
    outcomes = rng.integers(0, 6, size=(5, 3))
    states = np.stack([PAULI_EIGENSTATES[inputs[:, j]] for j in range(3)])
    povms = np.stack([PAULI_EIGENSTATES[outcomes[:, j]] for j in range(3)])

    old = build_lpdo(tn_data=chain_arrays)
    params, skeleton = qtn.pack(old.select("V"))
    states_tn, povm_tn = experiments_to_hyper_tn(np.stack([inputs, outcomes], axis=1), [PAULI_EIGENSTATES] * 3, [PAULI_EIGENSTATES] * 3)
    state_params, state_skeleton = qtn.pack(evolve_hyper_tn(states_tn, qc))
    povm_params, povm_skeleton = qtn.pack(povm_tn)
    _, (old_probs, old_nll, old_tp) = loss_fn(params, state_params, povm_params, (skeleton, state_skeleton, povm_skeleton))

    lpdo_tn = LPDO(Topology.chain(3), chain_arrays).network()
    probs = probabilities(lpdo_tn, states, povms, circuit_network(qc))
    np.testing.assert_allclose(probs, old_probs, rtol=1e-10)
    assert negative_log_likelihood(lpdo_tn, probs) == pytest.approx(float(old_nll), rel=1e-10)
    assert tp_penalty(lpdo_tn) == pytest.approx(float(old_tp), rel=1e-8)


def test_two_qubit_depolarizing_matches_legacy_even_layer():
    new = two_qubit_depolarizing(Topology.chain(4), 0.1, edges=[(0, 1), (2, 3)])
    old = dep2loc_even(0.1, 4)
    rows = [f"kn{j}" for j in range(4)] + [f"bn{j}" for j in range(4)]
    cols = [f"k{j}" for j in range(4)] + [f"b{j}" for j in range(4)]
    np.testing.assert_allclose(new.to_dense(), old.contract().to_dense(rows, cols), atol=1e-12)
