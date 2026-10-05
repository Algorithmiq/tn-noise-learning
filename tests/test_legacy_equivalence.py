"""Temporary checks of the new topology-generic code against the legacy 1D implementation."""

from __future__ import annotations

import numpy as np
import pytest

from tn_noise_learning.lpdo import LPDO
from tn_noise_learning.tn_core.lpdo_utils import build_lpdo, lpdo_fidelity, trace_lpdo
from tn_noise_learning.topology import Topology


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
