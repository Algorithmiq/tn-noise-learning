from __future__ import annotations

import numpy as np
import pytest

from tn_noise_learning.lpdo import LPDO
from tn_noise_learning.topology import Topology


@pytest.fixture(params=[Topology.chain(3), Topology.square(2, 2)], ids=["chain3", "square22"])
def topology(request):
    return request.param


def test_identity_trace(topology):
    assert LPDO.identity(topology).trace() == pytest.approx(2**topology.num_sites)


def test_distance_to_self_is_zero(topology):
    model = LPDO.random(topology, bond_dim=2, kraus_dim=2, seed=0)
    assert model.distance(model) == pytest.approx(0.0, abs=1e-12)
    assert model.distance(LPDO.identity(topology)) > 0


def test_random_trace_scale(topology):
    traces = [LPDO.random(topology, bond_dim=3, kraus_dim=2, seed=s).trace() for s in range(20)]
    assert 0.1 < np.mean(traces) / 2**topology.num_sites < 10


def test_network_round_trip(topology):
    model = LPDO.random(topology, bond_dim=2, kraus_dim=3, seed=1)
    rebuilt = LPDO.from_network(topology, model.kraus_network())
    for a, b in zip(model.tensors, rebuilt.tensors, strict=True):
        np.testing.assert_allclose(a, b)


def test_to_dense_identity():
    dense = LPDO.identity(Topology.chain(2)).to_dense()
    np.testing.assert_allclose(dense, np.eye(16))


def test_zarr_round_trip(tmp_path):
    model = LPDO.random(Topology.square(2, 3), bond_dim=2, kraus_dim=2, seed=3)
    model.save(tmp_path / "model.zarr")
    loaded = LPDO.load(tmp_path / "model.zarr")
    assert loaded.topology == model.topology
    for a, b in zip(model.tensors, loaded.tensors, strict=True):
        np.testing.assert_array_equal(a, b)


def test_shape_validation():
    with pytest.raises(ValueError, match="expected shape"):
        LPDO(Topology.chain(2), [np.zeros((2, 2, 2)), np.zeros((1, 2, 2))])
