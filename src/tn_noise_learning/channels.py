"""Noise channels as LPDOs on any topology, and the Pauli eigenstate basis used for preparation and measurement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .lpdo import LPDO

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .topology import Topology

PAULI = np.array(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
    dtype=complex,
)

PAULI_EIGENSTATES = np.array(
    [
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, -0.5], [-0.5, 0.5]],
        [[0.5, -0.5j], [0.5j, 0.5]],
        [[0.5, 0.5j], [-0.5j, 0.5]],
    ],
    dtype=complex,
)
"""States ``|0>, |1>, |+>, |−>, |+i>, |−i>``; effect ``2 * basis + bit`` is outcome ``bit`` of basis Z, X or Y."""


def pauli_eigenstates(num_sites: int) -> np.ndarray:
    """The six Pauli eigenstates on every site, shape ``(num_sites, 6, 2, 2)``."""
    return np.broadcast_to(PAULI_EIGENSTATES, (num_sites, 6, 2, 2)).copy()


def _place(block: np.ndarray, topology: Topology, site: int, partner: int | None = None) -> np.ndarray:
    """Reshape a ``(kraus, 2, 2)`` or ``(bond, kraus, 2, 2)`` block into canonical site axis order."""
    neighbors = topology.neighbors(site)
    bonds = [1] * len(neighbors)
    if partner is not None:
        bonds[neighbors.index(partner)] = block.shape[0]
        block = block.reshape(block.shape[1:]) if block.shape[0] == 1 else block
        return block.reshape((*bonds, *block.shape[-3:]))
    return block.reshape((*bonds, *block.shape))


def pauli_channel(topology: Topology, probs: Iterable[float] | np.ndarray) -> LPDO:
    """Independent single-qubit Pauli channels ``ρ → Σ_P p_P P ρ P``.

    Args:
        topology: site connectivity (all bonds have size one).
        probs: ``(p_I, p_X, p_Y, p_Z)`` shared by all sites, or one row per site with shape ``(num_sites, 4)``.
    """
    probs = np.broadcast_to(np.asarray(probs, dtype=float), (topology.num_sites, 4))
    if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0):
        raise ValueError("Pauli probabilities must be non-negative and sum to one on every site.")
    tensors = [_place(np.sqrt(p)[:, None, None] * PAULI, topology, j) for j, p in enumerate(probs)]
    return LPDO(topology, tensors)


def depolarizing(topology: Topology, p: float) -> LPDO:
    """Single-qubit depolarizing channel ``(1 − p) ρ + (p / 3) Σ_{P≠I} P ρ P`` on every site."""
    return pauli_channel(topology, [1 - p, p / 3, p / 3, p / 3])


def two_qubit_depolarizing(topology: Topology, p: float, edges: Iterable[tuple[int, int]] | None = None) -> LPDO:
    """Two-qubit depolarizing channel ``(1 − p) ρ + p · Tr[ρ] 𝟙/4`` on pairwise-disjoint edges, identity elsewhere.

    Args:
        topology: site connectivity.
        p: depolarizing probability; the sixteen two-qubit Paulis get weights ``1 − 15p/16`` (identity) and ``p/16``.
        edges: edges that carry the channel (default: all topology edges). They must be topology edges and
            must not share sites.
    """
    edges = tuple(topology.edges) if edges is None else tuple((min(e), max(e)) for e in edges)
    sites = [s for e in edges for s in e]
    if len(set(sites)) != len(sites):
        raise ValueError("Edges carrying the two-qubit channel must be pairwise disjoint.")
    if any(e not in topology.edges for e in edges):
        raise ValueError(f"Edges {edges} must all be edges of the topology.")
    weights = np.full((4, 4), p / 16.0)
    weights[0, 0] = 1 - 15 * p / 16.0
    first = np.einsum("am,mij->amij", np.eye(4), PAULI)
    second = np.einsum("am,mij->amij", np.sqrt(weights), PAULI)
    tensors = [_place(np.eye(2, dtype=complex)[None], topology, j) for j in range(topology.num_sites)]
    for i, j in edges:
        tensors[i] = _place(first, topology, i, partner=j)
        tensors[j] = _place(second, topology, j, partner=i)
    return LPDO(topology, tensors)
