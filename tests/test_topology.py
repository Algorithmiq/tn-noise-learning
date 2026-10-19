from __future__ import annotations

import pytest

from tn_noise_learning.topology import Topology


def test_chain_edges():
    assert Topology.chain(4).edges == ((0, 1), (1, 2), (2, 3))


def test_square_edges_row_major():
    assert Topology.square(2, 3).edges == ((0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5))


def test_from_edges_normalises_and_infers_sites():
    topo = Topology.from_edges([(2, 0), (0, 2), (1, 2)])
    assert topo.num_sites == 3
    assert topo.edges == ((0, 2), (1, 2))
    assert topo.neighbors(2) == (0, 1)
    assert topo.degree(1) == 1


@pytest.mark.parametrize("edges", [[(0, 0)], [(0, 5)]])
def test_invalid_edges_raise(edges):
    with pytest.raises(ValueError, match=r"not allowed|outside"):
        Topology(3, tuple(edges))
