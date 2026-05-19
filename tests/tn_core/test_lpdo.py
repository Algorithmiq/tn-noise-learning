"""Smoke tests for the LPDO data structures and helpers."""

from __future__ import annotations

import pytest

from tn_noise_learning.tn_core import (
    LPDOSuperOperator1D,
    build_lpdo,
    compose_lpdos,
    lpdo_fidelity,
    lpdo_to_mpo,
)


@pytest.fixture
def small_random_lpdo():
    return build_lpdo(num_qubits=2, virtual_bd=2, kraus_bd=2, init_method="normal_complex", seed=0)


def test_build_lpdo_returns_correct_type(small_random_lpdo):
    assert isinstance(small_random_lpdo, LPDOSuperOperator1D)
    assert small_random_lpdo.L == 2


def test_lpdo_to_mpo_runs(small_random_lpdo):
    mpo = lpdo_to_mpo(small_random_lpdo, ptm=True)
    assert mpo.L == 2


def test_compose_lpdos_identity_like(small_random_lpdo):
    composed = compose_lpdos(small_random_lpdo, small_random_lpdo)
    assert composed.L == 2


def test_lpdo_fidelity_self_is_zero(small_random_lpdo):
    # Fidelity-as-distance — comparing an LPDO with itself yields ~0.
    fid = lpdo_fidelity(small_random_lpdo, small_random_lpdo)
    assert fid < 1e-6
