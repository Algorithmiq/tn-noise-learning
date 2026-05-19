"""Pauli Transfer Matrix (PTM) conversion utilities.

Re-exports the core PTM constructors from `tn_core.mpo_utils` and `qpt.states`
for easier discovery, plus a couple of small helpers.
"""

from __future__ import annotations

import numpy as np

from ..qpt.states import rho_to_ptm
from ..tn_core.mpo_utils import unitary_to_ptm

__all__ = ["pauli_basis", "ptm_to_kraus_single_qubit", "rho_to_ptm", "unitary_to_ptm"]


def pauli_basis() -> np.ndarray:
    """Return the normalised single-qubit Pauli basis ``{I, X, Y, Z} / √2``."""
    return np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ]
    ) / np.sqrt(2)


def ptm_to_kraus_single_qubit(ptm: np.ndarray, tol: float = 1e-12) -> list[np.ndarray]:
    r"""Convert a single-qubit channel from PTM form into a (possibly redundant) list of Kraus operators.

    Uses the Choi-Jamiolkowski isomorphism: ``J = (1/d) Σ_{ij} PTM_{ij} (P_i^T ⊗ P_j)``,
    then eigen-decomposes ``J`` to read off the Kraus operators.

    Args:
        ptm: a 4×4 PTM.
        tol: eigenvalues below this magnitude are treated as zero.

    Returns:
        List of 2×2 Kraus operators.
    """
    if ptm.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 single-qubit PTM, got {ptm.shape}.")
    paulis = pauli_basis() * np.sqrt(2)
    d = 2

    choi = np.zeros((d * d, d * d), dtype=complex)
    for i in range(4):
        for j in range(4):
            choi += ptm[i, j] * np.kron(paulis[i].conj(), paulis[j]) / d

    eigvals, eigvecs = np.linalg.eigh(choi)
    kraus_ops = []
    for w, v in zip(eigvals, eigvecs.T, strict=True):
        if w > tol:
            kraus_ops.append(np.sqrt(w) * v.reshape(d, d))
    return kraus_ops
