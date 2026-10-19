# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qubit connectivity on which an LPDO places its virtual bonds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import quimb.tensor as qtn

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class Topology:
    """Graph of qubit sites ``0..num_sites-1`` and the edges that carry virtual bonds.

    Site integers are the qubit order of the dataset and of the Qiskit circuit. Edges are
    stored normalised as sorted ``(i, j)`` pairs with ``i < j``, without duplicates.

    Args:
        num_sites: number of qubits.
        edges: pairs of sites connected by a bond.
    """

    num_sites: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Validate sites and normalise the edge list."""
        if self.num_sites < 1:
            raise ValueError(f"num_sites must be positive, got {self.num_sites}.")
        normalised = set()
        for i, j in self.edges:
            if i == j:
                raise ValueError(f"Self-loop on site {i} is not allowed.")
            if not (0 <= i < self.num_sites and 0 <= j < self.num_sites):
                raise ValueError(f"Edge ({i}, {j}) is outside the site range 0..{self.num_sites - 1}.")
            normalised.add((min(i, j), max(i, j)))
        object.__setattr__(self, "edges", tuple(sorted(normalised)))

    @classmethod
    def chain(cls, num_sites: int) -> Topology:
        """Open 1D chain ``0 - 1 - ... - (num_sites - 1)``."""
        return cls(num_sites, tuple(qtn.edges_1d_chain(num_sites)))

    @classmethod
    def square(cls, rows: int, cols: int) -> Topology:
        """Open 2D square lattice, with site ``(r, c)`` mapped row-major to ``r * cols + c``."""
        edges = [(r1 * cols + c1, r2 * cols + c2) for (r1, c1), (r2, c2) in qtn.edges_2d_square(rows, cols)]
        return cls(rows * cols, tuple(edges))

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[int, int]], num_sites: int | None = None) -> Topology:
        """Custom topology from an edge list; ``num_sites`` defaults to the largest site index plus one."""
        pairs = tuple((int(i), int(j)) for i, j in edges)
        if num_sites is None:
            num_sites = 1 + max((max(e) for e in pairs), default=0)
        return cls(num_sites, pairs)

    def neighbors(self, site: int) -> tuple[int, ...]:
        """Sites bonded to ``site``, in ascending order."""
        return tuple(sorted(j if i == site else i for i, j in self.edges if site in (i, j)))

    def degree(self, site: int) -> int:
        """Number of bonds attached to ``site``."""
        return len(self.neighbors(site))
