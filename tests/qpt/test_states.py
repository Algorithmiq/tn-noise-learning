"""Smoke tests for QPT input-state and PTM helpers."""

from __future__ import annotations

import numpy as np

from tn_noise_learning.qpt import (
    ic_condition_check,
    pauli_eigenstates,
    rho_to_ptm,
    sic_tetrahedron_states,
)


def test_pauli_eigenstates_shape():
    rhos = pauli_eigenstates()
    assert rhos.shape == (6, 2, 2)


def test_pauli_eigenstates_are_ic():
    rhos = pauli_eigenstates()
    assert ic_condition_check(rhos)


def test_sic_tetrahedron_shape():
    states = sic_tetrahedron_states()
    assert states.shape == (4, 2, 2)


def test_rho_to_ptm_single_qubit_identity():
    ptm = rho_to_ptm(np.eye(2, dtype=complex) / 2)
    # |I⟩⟩ in the normalised Pauli basis: (1/√2) on the I-component only.
    expected = np.zeros((4,))
    expected[0] = 1 / np.sqrt(2)
    np.testing.assert_allclose(ptm, expected, atol=1e-12)
