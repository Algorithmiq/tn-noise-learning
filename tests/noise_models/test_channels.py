"""Smoke tests for the noise channel dispatcher and individual PTMs."""

from __future__ import annotations

import numpy as np
import pytest

from tn_noise_learning.noise_models import (
    amplitude_damping_ptm,
    dephasing_ptm,
    depolarizing_ptm,
    select_noise_channel,
)


def test_identity_returns_none():
    assert select_noise_channel("identity", num_qubits=2, tensor_type="mpo") is None


def test_even_depol_mpo_smoke():
    mpo = select_noise_channel("even_depol1_odd_depol05", num_qubits=4, tensor_type="mpo")
    assert mpo.L == 4


def test_even_depol_lpdo_smoke():
    lpdo = select_noise_channel("even_depol1_odd_depol05", num_qubits=4, tensor_type="lpdo")
    assert lpdo.L == 4


@pytest.mark.parametrize("ptm_fn", [dephasing_ptm, depolarizing_ptm, amplitude_damping_ptm])
def test_single_qubit_ptms_are_4x4(ptm_fn):
    ptm = ptm_fn(0.1)
    assert ptm.shape == (4, 4)


@pytest.mark.parametrize("ptm_fn", [dephasing_ptm, depolarizing_ptm])
def test_zero_strength_is_identity(ptm_fn):
    np.testing.assert_allclose(ptm_fn(0.0), np.eye(4))


def test_unknown_noise_raises():
    with pytest.raises(ValueError):
        select_noise_channel("definitely-not-a-preset", num_qubits=2)
