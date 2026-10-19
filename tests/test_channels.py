from __future__ import annotations

from functools import reduce

import numpy as np
import pytest

from tn_noise_learning.channels import PAULI, PAULI_EIGENSTATES, depolarizing, pauli_channel, two_qubit_depolarizing
from tn_noise_learning.topology import Topology


def kraus_sum(kraus_ops):
    return sum(np.kron(k, k.conj()) for k in kraus_ops)


def test_pauli_eigenstates_are_projective_bases():
    for basis in range(3):
        np.testing.assert_allclose(PAULI_EIGENSTATES[2 * basis] + PAULI_EIGENSTATES[2 * basis + 1], np.eye(2))
    for state in PAULI_EIGENSTATES:
        np.testing.assert_allclose(state @ state, state)


@pytest.mark.parametrize("topology", [Topology.chain(2), Topology.square(2, 2)], ids=["chain2", "square22"])
def test_depolarizing_matches_kraus_sum(topology):
    p = 0.3
    channel = depolarizing(topology, p)
    assert channel.trace() == pytest.approx(2**topology.num_sites)
    local = kraus_sum([np.sqrt(w) * P for w, P in zip([1 - p, p / 3, p / 3, p / 3], PAULI, strict=True)])
    expected = reduce(np.kron, [local] * topology.num_sites)
    # kron over sites interleaves (out_ket, out_bra) per site in the rows and (in_ket, in_bra) in the columns;
    # reorder to (all out kets, all out bras) x (all in kets, all in bras) as LPDO.to_dense does.
    n = topology.num_sites
    expected = expected.reshape([2] * 4 * n)
    order = [2 * s for s in range(n)] + [2 * s + 1 for s in range(n)]
    order += [2 * n + 2 * s for s in range(n)] + [2 * n + 2 * s + 1 for s in range(n)]
    expected = expected.transpose(order).reshape(4**n, 4**n)
    np.testing.assert_allclose(channel.to_dense(), expected, atol=1e-12)


def test_two_qubit_depolarizing_dense_formula():
    p = 0.25
    channel = two_qubit_depolarizing(Topology.chain(2), p)
    vec_identity = np.eye(4).reshape(-1)
    expected = (1 - p) * np.eye(16) + p * np.outer(vec_identity, vec_identity) / 4
    np.testing.assert_allclose(channel.to_dense(), expected, atol=1e-12)
    assert channel.trace() == pytest.approx(4.0)


def test_two_qubit_depolarizing_on_lattice_edges():
    topology = Topology.square(2, 2)
    channel = two_qubit_depolarizing(topology, 0.1, edges=[(0, 1), (2, 3)])
    assert channel.trace() == pytest.approx(16.0)
    assert channel.bond_size(0, 1) == 4
    assert channel.bond_size(0, 2) == 1


def test_two_qubit_depolarizing_validation():
    topology = Topology.square(2, 2)
    with pytest.raises(ValueError, match="disjoint"):
        two_qubit_depolarizing(topology, 0.1, edges=[(0, 1), (1, 3)])
    with pytest.raises(ValueError, match="edges of the topology"):
        two_qubit_depolarizing(topology, 0.1, edges=[(0, 3)])


def test_pauli_channel_validation():
    with pytest.raises(ValueError, match="sum to one"):
        pauli_channel(Topology.chain(2), [0.5, 0.5, 0.5, 0.5])
