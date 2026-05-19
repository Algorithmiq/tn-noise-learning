"""Smoke tests for PTM helpers."""

from __future__ import annotations

import numpy as np

from tn_noise_learning.utils import pauli_basis, ptm_to_kraus_single_qubit


def test_pauli_basis_orthonormality():
    paulis = pauli_basis()
    for i in range(4):
        for j in range(4):
            val = np.trace(paulis[i].conj().T @ paulis[j])
            expected = 1.0 if i == j else 0.0
            np.testing.assert_allclose(val, expected, atol=1e-12)


def test_ptm_to_kraus_identity_recovers_identity():
    kraus_ops = ptm_to_kraus_single_qubit(np.eye(4))
    assert len(kraus_ops) >= 1
    summed = sum(k.conj().T @ k for k in kraus_ops)
    np.testing.assert_allclose(summed, np.eye(2), atol=1e-10)
