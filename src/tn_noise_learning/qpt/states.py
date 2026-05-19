"""Reference input states for QPT (Pauli eigenstates, SIC tetrahedron) and PTM utilities."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from functools import reduce

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import DensityMatrix


def pauli_eigenstates(out: str = "density") -> NDArray:
    r"""Overcomplete set of 6 single-qubit Pauli eigenstates.

    Args:
        out: ``"density"`` (default) returns 2×2 density matrices,
            ``"kets"`` returns 2-component complex amplitudes.

    Returns:
        Array of shape ``(6, 2, 2)`` (density matrices) or ``(6, 2)`` (kets) in
        the order ``|0⟩, |1⟩, |+⟩, |−⟩, |+i⟩, |−i⟩``.
    """
    kets = {
        "0": np.array([1, 0], dtype=complex),
        "1": np.array([0, 1], dtype=complex),
        "+": (1 / np.sqrt(2) * np.array([1, 1], dtype=complex)),
        "-": (1 / np.sqrt(2) * np.array([1, -1], dtype=complex)),
        "+i": (1 / np.sqrt(2) * np.array([1, 1j], dtype=complex)),
        "-i": (1 / np.sqrt(2) * np.array([1, -1j], dtype=complex)),
    }

    rhos = {
        "0": np.array([[1, 0], [0, 0]], dtype=complex),
        "1": np.array([[0, 0], [0, 1]], dtype=complex),
        "+": 0.5 * np.array([[1, 1], [1, 1]], dtype=complex),
        "-": 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex),
        "+i": 0.5 * np.array([[1, -1j], [1j, 1]], dtype=complex),
        "-i": 0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex),
    }

    if out == "kets":
        return np.array(list(kets.values()))
    return np.array(list(rhos.values()))


def sic_tetrahedron_states(type: str = "pauli_symmetric") -> NDArray:
    r"""Minimal four-state SIC-POVM density matrices.

    Args:
        type: ``"pauli_symmetric"`` (default) uses the Bloch vectors of
            Eq. (2.6) in `Renes et al. <https://arxiv.org/abs/quant-ph/0405084v2>`_.
            ``"simple"`` places one vertex on ``|0⟩⟨0|``.
    """
    if type == "pauli_symmetric":
        bloch_coordinates = np.array(
            [
                [+1, +1, +1],
                [+1, -1, -1],
                [-1, +1, -1],
                [-1, -1, +1],
            ]
        ) / np.sqrt(3)
    elif type == "simple":
        bloch_coordinates = np.array(
            [
                [0, 0, 1],
                [+2 * np.sqrt(2) / 3, 0, -1 / 3],
                [-1 * np.sqrt(2) / 3, -np.sqrt(2 / 3), -1 / 3],
                [-1 * np.sqrt(2) / 3, +np.sqrt(2 / 3), -1 / 3],
            ]
        )
    else:
        raise ValueError(f"SIC-POVM {type} not implemented; only 'simple' and 'pauli_symmetric'.")

    id = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    pauli_mats = np.array([x, y, z])

    return np.array([(id + np.einsum("ijk,i->jk", pauli_mats, bv)) / 2 for bv in bloch_coordinates])


def state_to_bloch_vectors(rho: NDArray) -> NDArray:
    """Compute the Bloch vector(s) of one or more single-qubit density matrices."""
    rho = np.array(rho)
    if len(rho.shape) < 3:
        rho = rho[np.newaxis, :, :]

    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    return np.array([[np.trace(r @ p) for p in [x, y, z]] for r in rho])


def rho_to_ptm(rho: np.ndarray | DensityMatrix) -> NDArray[np.float64]:
    r"""Convert a density matrix to its vectorised PTM form.

    Args:
        rho: a ``(2**n, 2**n)`` matrix where ``n`` is the number of qubits.

    Returns:
        A ``(4,) * n``-shaped tensor whose entry ``(i_1, ..., i_n)`` is the
        expectation value of the corresponding tensor product of normalised
        Paulis.
    """
    Id = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    paulis = np.array([Id, X, Y, Z])

    if isinstance(rho, np.ndarray):
        num_qubits = int(np.log2(rho.shape[0]))
    if isinstance(rho, DensityMatrix):
        num_qubits = len(rho.dims())
        rho = rho.data

    result = np.empty((4,) * num_qubits)
    for idx in itertools.product(range(4), repeat=num_qubits):
        p_i = reduce(np.kron, [paulis[i] / np.sqrt(2) for i in idx])
        result[idx] = np.real(np.trace(p_i @ rho))
    return result


def labels_to_tensors(labels: Iterable[int], state_space: int = 6) -> NDArray:
    """One-hot encode a list of integer state-/POVM-labels.

    Example: with ``state_space = 3`` and ``labels = [0, 2, 0]`` the output is
    ``[[1, 0, 0], [0, 0, 1], [1, 0, 0]]``.

    Args:
        labels: integer labels, one per qubit.
        state_space: cardinality of the label set (6 for the Pauli OC set, 4
            for the SIC tetrahedron, 3 for the Z/X/Y measurement bases).
    """
    one_hot_matrix = np.zeros((len(labels), state_space))
    one_hot_matrix[range(len(labels)), labels] = 1
    return one_hot_matrix


def ic_condition_check(states: np.ndarray | list) -> bool:
    """Check the informational-completeness condition: the states sum to the identity (up to a scalar)."""
    states = np.array(states)

    if len(states.shape) == 3:
        state_sum = np.sum(states, axis=0)
        return np.count_nonzero(state_sum - np.diag(np.diagonal(state_sum))) == 0

    if len(states.shape) == 4:
        state_sum = np.sum(states, axis=1)
        return all(np.count_nonzero(_ - np.diag(np.diagonal(_))) == 0 for _ in state_sum)

    raise ValueError(
        f"Weird shape for states/povms: {states.shape}. Expected (num_qubits, num_states, 2, 2) or (num_states, 2, 2)."
    )
