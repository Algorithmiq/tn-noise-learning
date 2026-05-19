"""LPDO representations of quantum noise channels.

The LPDO form is convenient because it preserves complete positivity by
construction. Each channel is built as Kraus tensors that, when interleaved
with their conjugates, define a valid quantum operation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit

from ..tn_core.lpdo_model import LPDOSuperOperator1D
from ..tn_core.lpdo_utils import build_lpdo, compose_lpdos
from .mpo_channels import parameterised_unitary
from .sparse_pauli_lindblad import HADAMARD_WALSH_1Q, HADAMARD_WALSH_2Q, parse_spl_coefficients


def single_qubit_coherent_errors(params: np.ndarray) -> LPDOSuperOperator1D:
    """Build an LPDO whose Kraus operators are tensor products of single-qubit unitaries.

    Args:
        params: array of shape ``(num_qubits, 3)`` — per-qubit parameters in
            the same convention as
            [`parameterised_unitary`][tn_noise_learning.noise_models.mpo_channels.parameterised_unitary].
    """
    if isinstance(params, list):
        params = np.array(params)

    if params.shape[1] != 3:
        raise ValueError(f"Must specify 3 parameters per qubit, got instead {params.shape}.")

    num_qubits = params.shape[0]
    single_qubit_kraus = [parameterised_unitary(p) for p in params]

    for n, op in enumerate(single_qubit_kraus):
        if n == 0 or n == num_qubits - 1:
            single_qubit_kraus[n] = op.reshape(1, 1, 2, 2)
        else:
            single_qubit_kraus[n] = op.reshape(1, 1, 1, 2, 2)

    return build_lpdo(tn_data=single_qubit_kraus)


def two_qubit_depolarizing_error(p: float, num_qubits: int = 2, target_qubits: Sequence[int] = (0, 1)) -> list[np.ndarray]:
    r"""Kraus operators for an LPDO of a two-local depolarising channel.

    The two-qubit depolarising channel has the form
    ``D[ρ] = (1 - p) ρ + (p / 15) Σ_{(P_1, P_2) ≠ (I, I)} P_1 P_2 ρ P_1 P_2``.
    """
    id = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    p_id = np.sqrt(1.0 - 15.0 * p / 16.0)
    p_op = np.sqrt(p / 16.0)

    alpha = np.sqrt(p_id - p_op)
    beta = np.sqrt(p_op)

    depol_ops = np.array(
        [[alpha * id, 0 * id, 0 * id, 0 * id], [beta * id, beta * x, beta * y, beta * z]]
    )

    end_tn = np.array([[id]])
    mid_tn = np.array([[[id]]])
    kraus_ops = [end_tn] + [mid_tn] * (num_qubits - 2) + [end_tn]

    q0, q1 = min(target_qubits), max(target_qubits)
    if abs(q0 - q1) != 1:
        raise TypeError(f"Qubits must be adjacent, got target_qubits = {target_qubits}.")

    if q0 == 0:
        kraus_ops[q0] = depol_ops
        if num_qubits == 2:
            kraus_ops[q1] = depol_ops
        else:
            kraus_ops[q1] = np.expand_dims(depol_ops, 1)
    elif q1 == num_qubits - 1:
        kraus_ops[q0] = np.expand_dims(depol_ops, 0)
        kraus_ops[q1] = depol_ops
    else:
        kraus_ops[q0] = np.expand_dims(depol_ops, 0)
        kraus_ops[q1] = np.expand_dims(depol_ops, 1)

    return kraus_ops


def dep2loc_even(p: float, num_qubits: int) -> LPDOSuperOperator1D:
    """LPDO of two-local depolarising channels on the even pairs ``(0, 1), (2, 3), …``."""
    noise_lpdo = build_lpdo(tn_data=QuantumCircuit(num_qubits))
    for qub in range(0, num_qubits - 1, 2):
        kraus_ops = two_qubit_depolarizing_error(p, num_qubits=num_qubits, target_qubits=[qub, qub + 1])
        dep = build_lpdo(tn_data=kraus_ops)
        noise_lpdo = compose_lpdos(dep, noise_lpdo)
    return noise_lpdo


def dep2loc_odd(p: float, num_qubits: int) -> LPDOSuperOperator1D:
    """LPDO of two-local depolarising channels on the odd pairs ``(1, 2), (3, 4), …``."""
    noise_lpdo = build_lpdo(tn_data=QuantumCircuit(num_qubits))
    for qub in range(1, num_qubits - 1, 2):
        kraus_ops = two_qubit_depolarizing_error(p, num_qubits=num_qubits, target_qubits=[qub, qub + 1])
        dep = build_lpdo(tn_data=kraus_ops)
        noise_lpdo = compose_lpdos(dep, noise_lpdo)
    return noise_lpdo


def one_qubit_pauli_channel(coeffs: np.ndarray, num_qubits: int = 2, target_qubits: Sequence[int] = (0, 1)) -> list[np.ndarray]:
    """Kraus operators of a tensor product of single-qubit Pauli channels.

    Args:
        coeffs: shape ``(num_qubits, 4)`` matrix of per-qubit Kraus probabilities
            ``(p_I, p_X, p_Y, p_Z)``; each row must sum to one.
    """
    if not np.all(np.isclose(np.sum(coeffs, axis=1), 1)):
        raise ValueError("Probabilities do not sum to one.")

    num_qubits = len(coeffs)

    id = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    paulis = np.array([id, x, y, z])

    kraus_ops = []
    for n, p in enumerate(coeffs):
        local_kraus = np.expand_dims(np.einsum("i,ijk->ijk", np.sqrt(p), paulis), 0)
        if 0 < n < num_qubits - 1:
            local_kraus = np.expand_dims(local_kraus, 0)
        kraus_ops.append(local_kraus)
    return kraus_ops


def two_qubit_pauli_channel(coeffs: np.ndarray, num_qubits: int = 2, target_qubits: Sequence[int] = (0, 1)) -> list[np.ndarray]:
    r"""Kraus operators of a two-qubit Pauli channel acting on neighbouring qubits.

    Args:
        coeffs: 16-element array of Kraus probabilities in lexicographic order
            ``[c_II, c_IX, c_IY, c_IZ, c_XI, …, c_ZZ]``; must sum to one.
    """
    if not np.isclose(np.sum(coeffs), 1):
        raise ValueError("Probabilities do not sum to one.")

    id = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    c_ii, c_ix, c_iy, c_iz = np.sqrt(coeffs[0:4])
    c_xi, c_xx, c_xy, c_xz = np.sqrt(coeffs[4:8])
    c_yi, c_yx, c_yy, c_yz = np.sqrt(coeffs[8:12])
    c_zi, c_zx, c_zy, c_zz = np.sqrt(coeffs[12:16])

    loc_ops_a = np.array(
        [
            [id, 0 * id, 0 * id, 0 * id],
            [0 * id, x, 0 * id, 0 * id],
            [0 * id, 0 * id, y, 0 * id],
            [0 * id, 0 * id, 0 * id, z],
        ]
    )
    loc_ops_b = np.array(
        [
            [c_ii * id, c_ix * x, c_iy * y, c_iz * z],
            [c_xi * id, c_xx * x, c_xy * y, c_xz * z],
            [c_yi * id, c_yx * x, c_yy * y, c_yz * z],
            [c_zi * id, c_zx * x, c_zy * y, c_zz * z],
        ]
    )

    end_tn = np.array([[id]])
    mid_tn = np.array([[[id]]])
    kraus_ops = [end_tn] + [mid_tn] * (num_qubits - 2) + [end_tn]

    q0, q1 = min(target_qubits), max(target_qubits)
    if abs(q0 - q1) != 1:
        raise TypeError(f"Qubits must be adjacent, got target_qubits = {target_qubits}.")

    if q0 == 0:
        kraus_ops[q0] = loc_ops_a
        if num_qubits == 2:
            kraus_ops[q1] = loc_ops_b
        else:
            kraus_ops[q1] = np.expand_dims(loc_ops_b, 1)
    elif q1 == num_qubits - 1:
        kraus_ops[q0] = np.expand_dims(loc_ops_a, 0)
        kraus_ops[q1] = loc_ops_b
    else:
        kraus_ops[q0] = np.expand_dims(loc_ops_a, 0)
        kraus_ops[q1] = np.expand_dims(loc_ops_b, 1)

    return kraus_ops


def pauli_lindblad_coeffs_to_kraus_probs(
    filename: str, num_qubits: int, max_qubits: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read SPL coefficients from a file and return single- and two-qubit Kraus probabilities.

    The Hadamard-Walsh transform converts the Pauli-Lindblad coefficients into
    a Kraus probability vector on each site / link. Tiny negative entries from
    numerical noise are clipped to zero; each distribution is then renormalised.
    """
    single_coeffs, two_coeffs = parse_spl_coefficients(filename, num_qubits, max_qubits)

    single_qubit_probs = []
    for x, y, z in single_coeffs:
        vec = np.array(
            [
                [1.0],
                [np.exp(-2.0 * y - 2.0 * z)],
                [np.exp(-2.0 * x - 2.0 * z)],
                [np.exp(-2.0 * x - 2.0 * y)],
            ]
        )
        distribution = HADAMARD_WALSH_1Q @ vec
        distribution = 0.5 * (np.abs(distribution) + distribution)
        distribution = distribution / np.sum(distribution)
        single_qubit_probs.append(distribution.reshape(4))

    two_qubit_probs = []
    for c_xx, c_xy, c_xz, c_yx, c_yy, c_yz, c_zx, c_zy, c_zz in two_coeffs:
        vec = np.array(
            [
                [1.0],
                [np.exp(-2.0 * (c_xy + c_xz + c_yy + c_yz + c_zy + c_zz))],
                [np.exp(-2.0 * (c_xx + c_xz + c_yx + c_yz + c_zx + c_zz))],
                [np.exp(-2.0 * (c_xx + c_xy + c_yx + c_yy + c_zx + c_zy))],
                [np.exp(-2.0 * (c_yx + c_yy + c_yz + c_zx + c_zy + c_zz))],
                [np.exp(-2.0 * (c_xy + c_xz + c_yx + c_zx))],
                [np.exp(-2.0 * (c_xx + c_xz + c_yy + c_zy))],
                [np.exp(-2.0 * (c_xx + c_xy + c_yz + c_zz))],
                [np.exp(-2.0 * (c_xx + c_xy + c_xz + c_zx + c_zy + c_zz))],
                [np.exp(-2.0 * (c_xx + c_yy + c_yz + c_zx))],
                [np.exp(-2.0 * (c_xy + c_yx + c_yz + c_zy))],
                [np.exp(-2.0 * (c_xz + c_yx + c_yy + c_zz))],
                [np.exp(-2.0 * (c_xx + c_xy + c_xz + c_yx + c_yy + c_yz))],
                [np.exp(-2.0 * (c_xx + c_yx + c_zy + c_zz))],
                [np.exp(-2.0 * (c_xy + c_yy + c_zx + c_zz))],
                [np.exp(-2.0 * (c_xz + c_yz + c_zx + c_zy))],
            ]
        )
        distribution_2 = HADAMARD_WALSH_2Q @ vec
        distribution_2 = 0.5 * (np.abs(distribution_2) + distribution_2)
        distribution_2 = distribution_2 / np.sum(distribution_2)
        two_qubit_probs.append(distribution_2.reshape(16))

    return single_qubit_probs, two_qubit_probs


def sparse_pauli_lindblad_lpdo_from_file(filename: str, num_qubits: int, max_qubits: int) -> LPDOSuperOperator1D:
    """Build the LPDO of a sparse Pauli-Lindblad noise model from a coefficient file."""
    single_qub_probs, two_qub_probs = pauli_lindblad_coeffs_to_kraus_probs(filename, num_qubits, max_qubits)

    single_pauli_lpdo = build_lpdo(tn_data=one_qubit_pauli_channel(single_qub_probs))

    pauli_lindblad_lpdo = single_pauli_lpdo.copy()
    for n, p in enumerate(two_qub_probs):
        kraus_ops = two_qubit_pauli_channel(p, num_qubits, target_qubits=[n, n + 1])
        two_qub_lpdo = build_lpdo(tn_data=kraus_ops)
        pauli_lindblad_lpdo = compose_lpdos(pauli_lindblad_lpdo, two_qub_lpdo)

    return pauli_lindblad_lpdo
