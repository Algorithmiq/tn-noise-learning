"""Locally purified density operator (LPDO) representation of a quantum channel.

Site ``j`` holds a Kraus tensor ``V_j`` with axes ``(*bonds to neighbours in ascending order, kraus, out, in)``.
The channel is ``Λ(ρ) = Σ_kr V ρ V†``; the conjugate half of the network is never stored, it is generated from
the Kraus half by [`full_network`][tn_noise_learning.lpdo.full_network] with the index names below.

| leg               | Kraus half | conjugate half |
|-------------------|------------|----------------|
| bond on edge (i,j)| ``v{i}_{j}``| ``w{i}_{j}``  |
| kraus (shared)    | ``kr{j}``  | ``kr{j}``      |
| channel output    | ``kn{j}``  | ``bn{j}``      |
| channel input     | ``k{j}``   | ``b{j}``       |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
import zarr

from ._io import read_array, read_attr
from .topology import Topology

if TYPE_CHECKING:
    from pathlib import Path

KRAUS_TAG = "V"
DAGGER_TAG = "Vdag"
MAX_DENSE_SITES = 8


def site_tag(site: int) -> str:
    """Tag shared by the two tensors of ``site``."""
    return f"I{site}"


def bond_ind(i: int, j: int, *, dagger: bool = False) -> str:
    """Virtual bond index on edge ``(i, j)``."""
    a, b = min(i, j), max(i, j)
    return f"{'w' if dagger else 'v'}{a}_{b}"


def kraus_ind(site: int) -> str:
    """Kraus index shared by ``V_j`` and its conjugate."""
    return f"kr{site}"


def output_ind(site: int, *, dagger: bool = False) -> str:
    """Channel output (ket side, or bra side when ``dagger``)."""
    return f"{'bn' if dagger else 'kn'}{site}"


def input_ind(site: int, *, dagger: bool = False) -> str:
    """Channel input (ket side, or bra side when ``dagger``)."""
    return f"{'b' if dagger else 'k'}{site}"


def kraus_inds(topology: Topology, site: int) -> tuple[str, ...]:
    """Index names of ``V_site`` in canonical axis order."""
    return (*(bond_ind(i, site) for i in topology.neighbors(site)), kraus_ind(site), output_ind(site), input_ind(site))


def dagger_rename(topology: Topology) -> dict[str, str]:
    """Index renaming that turns the Kraus half into the conjugate half."""
    rename = {bond_ind(i, j): bond_ind(i, j, dagger=True) for i, j in topology.edges}
    for j in range(topology.num_sites):
        rename[output_ind(j)] = output_ind(j, dagger=True)
        rename[input_ind(j)] = input_ind(j, dagger=True)
    return rename


def full_network(kraus_tn: qtn.TensorNetwork, topology: Topology) -> qtn.TensorNetwork:
    """Join a Kraus network with its conjugate; works on JAX tracers inside a jitted loss."""
    dagger = kraus_tn.H.reindex(dagger_rename(topology)).retag({KRAUS_TAG: DAGGER_TAG})
    return qtn.TensorNetwork([kraus_tn, dagger])


@dataclass
class LPDO:
    """Quantum channel as an LPDO on a topology.

    Args:
        topology: site connectivity.
        tensors: one Kraus tensor per site in canonical axis order
            ``(*bonds to neighbours ascending, kraus, out, in)``.
    """

    topology: Topology
    tensors: list[np.ndarray]

    def __post_init__(self) -> None:
        """Validate tensor count, axis counts, physical dimensions and bond sizes."""
        if len(self.tensors) != self.topology.num_sites:
            raise ValueError(f"Expected {self.topology.num_sites} tensors, got {len(self.tensors)}.")
        self.tensors = [np.asarray(t) for t in self.tensors]
        for j, t in enumerate(self.tensors):
            expected = self.topology.degree(j) + 3
            if t.ndim != expected or t.shape[-2:] != (2, 2):
                raise ValueError(
                    f"Site {j}: expected shape (*{self.topology.degree(j)} bonds, kraus, 2, 2), got {t.shape}."
                )
        for i, j in self.topology.edges:
            if self.bond_size(i, j) != self.bond_size(j, i):
                raise ValueError(f"Bond sizes on edge ({i}, {j}) disagree.")

    @property
    def num_sites(self) -> int:
        """Number of qubits."""
        return self.topology.num_sites

    @property
    def kraus_dim(self) -> int:
        """Largest Kraus dimension over sites."""
        return max(t.shape[-3] for t in self.tensors)

    @property
    def bond_dim(self) -> int:
        """Largest virtual bond dimension (1 when there are no edges)."""
        return max((self.bond_size(i, j) for i, j in self.topology.edges), default=1)

    def bond_size(self, site: int, neighbor: int) -> int:
        """Size of the bond leg of ``site`` pointing at ``neighbor``."""
        return self.tensors[site].shape[self.topology.neighbors(site).index(neighbor)]

    @classmethod
    def random(cls, topology: Topology, bond_dim: int, kraus_dim: int, seed: int | None = None) -> LPDO:
        """Complex Gaussian Kraus tensors scaled so that the expected trace is ``2**num_sites``."""
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (4.0 * kraus_dim * bond_dim ** (len(topology.edges) / topology.num_sites)))
        tensors = []
        for j in range(topology.num_sites):
            shape = (*(bond_dim,) * topology.degree(j), kraus_dim, 2, 2)
            tensors.append(scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2))
        return cls(topology, tensors)

    @classmethod
    def identity(cls, topology: Topology) -> LPDO:
        """Identity channel: one Kraus operator ``𝟙`` per site, all bonds of size one."""
        return cls(
            topology,
            [
                np.eye(2, dtype=complex).reshape((1,) * (topology.degree(j) + 1) + (2, 2))
                for j in range(topology.num_sites)
            ],
        )

    @classmethod
    def from_network(cls, topology: Topology, kraus_tn: qtn.TensorNetwork) -> LPDO:
        """Rebuild from a Kraus network with the index names of [`kraus_inds`][tn_noise_learning.lpdo.kraus_inds]."""
        return cls(
            topology,
            [
                np.asarray(kraus_tn[site_tag(j)].transpose(*kraus_inds(topology, j)).data)
                for j in range(topology.num_sites)
            ],
        )

    def kraus_network(self) -> qtn.TensorNetwork:
        """Tensor network of the Kraus half, tagged ``I{j}`` and ``V``."""
        return qtn.TensorNetwork(
            [
                qtn.Tensor(t, inds=kraus_inds(self.topology, j), tags=(site_tag(j), KRAUS_TAG))
                for j, t in enumerate(self.tensors)
            ]
        )

    def network(self) -> qtn.TensorNetwork:
        """Full LPDO network with open legs ``kn, k`` (ket) and ``bn, b`` (bra) per site."""
        return full_network(self.kraus_network(), self.topology)

    def trace(self) -> float:
        """Trace of the channel's Choi-like network, ``2**num_sites`` for a trace-preserving channel."""
        connect = {output_ind(j): output_ind(j, dagger=True) for j in range(self.num_sites)}
        connect |= {input_ind(j): input_ind(j, dagger=True) for j in range(self.num_sites)}
        return float(np.real(self.network().reindex(connect).contract(optimize="auto-hq")))

    def distance(self, other: LPDO) -> float:
        """Squared normalised Frobenius distance ``‖Λ − Λ'‖_F² / 4**num_sites``."""
        return float(qtn.tensor_network_distance(self.network(), other.network()) / 2**self.num_sites) ** 2

    def to_dense(self) -> np.ndarray:
        """Superoperator matrix ``Σ_m K_m ⊗ conj(K_m)`` with site 0 most significant (small systems only)."""
        if self.num_sites > MAX_DENSE_SITES:
            raise ValueError(f"to_dense is limited to {MAX_DENSE_SITES} sites, got {self.num_sites}.")
        sites = range(self.num_sites)
        rows = [output_ind(j) for j in sites] + [output_ind(j, dagger=True) for j in sites]
        cols = [input_ind(j) for j in sites] + [input_ind(j, dagger=True) for j in sites]
        return np.asarray(self.network().contract(optimize="auto-hq").to_dense(rows, cols))

    def to_group(self, group: zarr.Group) -> None:
        """Write topology and tensors into an open zarr group."""
        group.attrs["num_sites"] = self.num_sites
        group.attrs["edges"] = [list(e) for e in self.topology.edges]
        for j, t in enumerate(self.tensors):
            group.create_array(f"site_{j}", shape=t.shape, dtype=t.dtype, overwrite=True)[:] = t

    @classmethod
    def from_group(cls, group: zarr.Group) -> LPDO:
        """Read an LPDO written by [`to_group`][tn_noise_learning.lpdo.LPDO.to_group]."""
        edges = tuple((int(i), int(j)) for i, j in read_attr(group, "edges"))
        topology = Topology(int(read_attr(group, "num_sites")), edges)
        return cls(topology, [read_array(group, f"site_{j}") for j in range(topology.num_sites)])

    def save(self, path: str | Path) -> None:
        """Save to a zarr store at ``path``."""
        self.to_group(zarr.open_group(str(path), mode="w"))

    @classmethod
    def load(cls, path: str | Path) -> LPDO:
        """Load from a zarr store written by [`save`][tn_noise_learning.lpdo.LPDO.save]."""
        return cls.from_group(zarr.open_group(str(path), mode="r"))
