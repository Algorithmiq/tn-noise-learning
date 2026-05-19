"""Smoke tests for the MPO construction helpers."""

from __future__ import annotations

import numpy as np
import qiskit as qk

from tn_noise_learning.tn_core.mpo_utils import (
    cnot_to_mpo,
    cnot_to_mpo_ptm,
    identity_mpo_ptm,
    qc_to_mpo,
    unitary_to_ptm,
)


def test_identity_mpo_ptm_shape():
    mpo = identity_mpo_ptm(3)
    assert mpo.L == 3


def test_qc_to_mpo_matches_qiskit_for_small_circuit():
    qc = qk.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    # qc_to_mpo internally verifies against Qiskit when num_qubits < 9.
    mpo = qc_to_mpo(qc)
    assert mpo.L == 2


def test_cnot_to_mpo_returns_two_qubit_object():
    mpo = cnot_to_mpo(2, 0, 1)
    assert mpo.L == 2


def test_cnot_to_mpo_ptm_returns_two_qubit_object():
    mpo = cnot_to_mpo_ptm(0, 1, 2)
    assert mpo.L == 2


def test_unitary_to_ptm_is_identity_on_identity():
    ptm = unitary_to_ptm(np.eye(2))
    np.testing.assert_allclose(ptm, np.eye(4), atol=1e-12)
