"""MPO representations of quantum noise channels in PTM form.

Every channel here lives on 4-D physical sites (the Pauli Transfer Matrix
representation), so the resulting MPOs can be applied directly to other
PTM-form MPOs/LPDOs in the QPT pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import quimb.tensor as qtn

from .sparse_pauli_lindblad import parse_spl_coefficients


def parameterised_unitary(params) -> np.ndarray:
    r"""Single-qubit unitary parameterisation from Eq. (19) of `Spengler et al. <https://arxiv.org/abs/1103.3408>`_.

    For a Haar-uniform unitary, sample
    ``params[1], params[2] ∼ U[0, 2π]`` and ``params[0] = arcsin(√x)`` with
    ``x ∼ U[0, 1]``.
    """
    return np.array(
        [
            [+np.exp(+1j * params[1]) * np.cos(params[0]), +np.exp(+1j * params[2]) * np.sin(params[0])],
            [-np.exp(-1j * params[2]) * np.sin(params[0]), +np.exp(-1j * params[1]) * np.cos(params[0])],
        ]
    )


# ---------------------------------------------------------------------------
# Single-qubit PTMs
# ---------------------------------------------------------------------------


def dephasing_ptm(p: float) -> np.ndarray:
    r"""PTM of a single-qubit dephasing channel ``N[ρ] = (1 - p) ρ + p Z ρ Z``."""
    if p > 1 or p < 0:
        raise ValueError(f"Noise strength p must be in [0, 1], got {p}.")
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1 - p, 0, 0],
            [0, 0, 1 - p, 0],
            [0, 0, 0, 1],
        ]
    )


def amplitude_damping_ptm(gamma: float) -> np.ndarray:
    r"""PTM of a single-qubit amplitude-damping channel with rate ``γ``."""
    if gamma > 1 or gamma < 0:
        raise ValueError(f"Noise strength gamma must be in [0, 1], got {gamma}.")
    r = np.sqrt(1 - gamma)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, r, 0, 0],
            [0, 0, r, 0],
            [gamma, 0, 0, 1 - gamma],
        ]
    )


def depolarizing_ptm(p: float) -> np.ndarray:
    r"""PTM of a single-qubit depolarising channel ``N[ρ] = (1 - p) ρ + p · 𝟙 / 2``."""
    if p > 1 or p < 0:
        raise ValueError(f"Noise strength p must be in [0, 1], got {p}.")
    r = 1 - p
    return np.array(
        [
            [1, 0, 0, 0],
            [0, r, 0, 0],
            [0, 0, r, 0],
            [0, 0, 0, r],
        ]
    )


def pauli_channel_ptm(p: np.ndarray) -> np.ndarray:
    r"""PTM of a single-qubit Pauli channel ``N[ρ] = p_0 ρ + p_x X ρ X + p_y Y ρ Y + p_z Z ρ Z``."""
    if (p > 1).any() or (p < 0).any():
        raise ValueError(f"Noise vector [px, py, pz] must be probabilities in [0, 1], got {p}.")
    p0 = 1 - np.sum(p)
    p1 = p0 + p[1] - p[2] - p[3]
    p2 = p0 - p[1] + p[2] - p[3]
    p3 = p0 - p[1] - p[2] + p[3]
    return np.array(
        [
            [1, 0, 0, 0],
            [0, p1, 0, 0],
            [0, 0, p2, 0],
            [0, 0, 0, p3],
        ]
    )


# ---------------------------------------------------------------------------
# Multi-qubit MPO noise channels
# ---------------------------------------------------------------------------


def single_qubit_coherent_error(params) -> qtn.MatrixProductOperator:
    """MPO of a tensor product of single-qubit coherent unitary errors (PTM form)."""
    pauli_matrices = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ]

    single_qubit_ptms = []
    for p in params:
        ptm = np.eye(4)
        u1 = parameterised_unitary(p)
        for i in range(1, 4):
            for j in range(1, 4):
                ptm[i, j] = 0.5 * np.real(
                    np.trace(np.dot(pauli_matrices[i], np.dot(u1, np.dot(pauli_matrices[j], np.conjugate(u1.T)))))
                )
        single_qubit_ptms.append(ptm)
    return qtn.MPO_product_operator(single_qubit_ptms)


def dep1loc_layer_mpo(depol_strength: float | Iterable[float], num_qubits: int) -> qtn.MatrixProductOperator:
    """Layer of single-qubit depolarising channels (uniform or per-site)."""
    if isinstance(depol_strength, float):
        single_qubit_noise = depolarizing_ptm(depol_strength)
        multi_qubit_noise = np.repeat(single_qubit_noise[np.newaxis, :, :], num_qubits, axis=0)
        return qtn.MPO_product_operator(multi_qubit_noise)
    if isinstance(depol_strength, Iterable):
        multi_qubit_noise = np.array([depolarizing_ptm(p) for p in depol_strength])
        return qtn.MPO_product_operator(multi_qubit_noise)
    raise ValueError(f"depol_strength must be a float or Iterable, got {type(depol_strength)}.")


def dep2loc_mpo(p: float, q1: int, q2: int, num_qubits: int) -> qtn.MatrixProductOperator:
    r"""MPO of a two-local depolarising channel on qubits ``q1`` and ``q2``.

    The channel is ``D[ρ] = (1 - p) ρ + p · (𝟙 / 2) ⊗ (𝟙 / 2)`` on the target pair.
    """
    iden_mat = np.eye(4)
    zero_mat = 0 * iden_mat
    proj_mat = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    target_qubit_tensors = np.array([np.sqrt(1 - p) * iden_mat, np.sqrt(p) * proj_mat])
    trivial_qubit_tensors = np.array(
        [
            [iden_mat, zero_mat],
            [zero_mat, iden_mat],
        ]
    )

    tensors_data: list[np.ndarray] = []
    if q1 < q2:
        for q in range(q1):
            tensors_data.append(iden_mat.reshape(1, 4, 4) if q == 0 else iden_mat.reshape(1, 1, 4, 4))
        tensors_data.append(target_qubit_tensors.reshape(2, 4, 4) if q1 == 0 else target_qubit_tensors.reshape(1, 2, 4, 4))
        for _q in range(q1 + 1, q2):
            tensors_data.append(trivial_qubit_tensors)
        tensors_data.append(
            target_qubit_tensors.reshape(2, 4, 4) if q2 == num_qubits - 1 else target_qubit_tensors.reshape(2, 1, 4, 4)
        )
        for q in range(q2 + 1, num_qubits):
            tensors_data.append(iden_mat.reshape(1, 4, 4) if q == num_qubits - 1 else iden_mat.reshape(1, 1, 4, 4))
    if q1 > q2:
        for q in range(q2):
            tensors_data.append(iden_mat.reshape(1, 4, 4) if q == 0 else iden_mat.reshape(1, 1, 4, 4))
        tensors_data.append(target_qubit_tensors.reshape(2, 4, 4) if q2 == 0 else target_qubit_tensors.reshape(1, 2, 4, 4))
        for _q in range(q2 + 1, q1):
            tensors_data.append(trivial_qubit_tensors)
        tensors_data.append(
            target_qubit_tensors.reshape(2, 4, 4) if q1 == num_qubits - 1 else target_qubit_tensors.reshape(2, 1, 4, 4)
        )
        for q in range(q1 + 1, num_qubits):
            tensors_data.append(iden_mat.reshape(1, 4, 4) if q == num_qubits - 1 else iden_mat.reshape(1, 1, 4, 4))

    return qtn.MatrixProductOperator(tensors_data)


def dep2loc_mpo_even(p: float, num_qubits: int) -> qtn.MatrixProductOperator:
    """MPO of two-local depolarising channels on the even pairs ``(0, 1), (2, 3), …``."""
    dep = qtn.MPO_identity(num_qubits, phys_dim=4)
    for i_pair in range(num_qubits // 2):
        dep = dep.apply(dep2loc_mpo(p, 2 * i_pair, 2 * i_pair + 1, num_qubits), compress=False)
    dep.permute_arrays(shape="lrud")
    return dep


def dep2loc_mpo_odd(p: float, num_qubits: int) -> qtn.MatrixProductOperator:
    """MPO of two-local depolarising channels on the odd pairs ``(1, 2), (3, 4), …``."""
    dep = qtn.MPO_identity(num_qubits, phys_dim=4)
    for i_pair in range(int(np.ceil(num_qubits / 2)) - 1):
        dep = dep.apply(dep2loc_mpo(p, 2 * i_pair + 1, 2 * i_pair + 2, num_qubits), compress=False)
    dep.permute_arrays(shape="lrud")
    return dep


# ---------------------------------------------------------------------------
# Sparse Pauli-Lindblad MPO
# ---------------------------------------------------------------------------


def _two_local_pauli_lindblad_block(coeffs: tuple[float, ...]) -> np.ndarray:
    """Build the 4×4×4×4 PTM tensor block for one two-qubit SPL link."""
    c_xx, c_xy, c_xz, c_yx, c_yy, c_yz, c_zx, c_zy, c_zz = coeffs
    return np.array(
        [
            np.diag(
                [
                    1.0,
                    np.exp(-2.0 * (c_xy + c_xz + c_yy + c_yz + c_zy + c_zz)),
                    np.exp(-2.0 * (c_xx + c_xz + c_yx + c_yz + c_zx + c_zz)),
                    np.exp(-2.0 * (c_xx + c_xy + c_yx + c_yy + c_zx + c_zy)),
                ]
            ),
            np.diag(
                [
                    np.exp(-2.0 * (c_yx + c_yy + c_yz + c_zx + c_zy + c_zz)),
                    np.exp(-2.0 * (c_xy + c_xz + c_yx + c_zx)),
                    np.exp(-2.0 * (c_xx + c_xz + c_yy + c_zy)),
                    np.exp(-2.0 * (c_xx + c_xy + c_yz + c_zz)),
                ]
            ),
            np.diag(
                [
                    np.exp(-2.0 * (c_xx + c_xy + c_xz + c_zx + c_zy + c_zz)),
                    np.exp(-2.0 * (c_xx + c_yy + c_yz + c_zx)),
                    np.exp(-2.0 * (c_xy + c_yx + c_yz + c_zy)),
                    np.exp(-2.0 * (c_xz + c_yx + c_yy + c_zz)),
                ]
            ),
            np.diag(
                [
                    np.exp(-2.0 * (c_xx + c_xy + c_xz + c_yx + c_yy + c_yz)),
                    np.exp(-2.0 * (c_xx + c_yx + c_zy + c_zz)),
                    np.exp(-2.0 * (c_xy + c_yy + c_zx + c_zz)),
                    np.exp(-2.0 * (c_xz + c_yz + c_zx + c_zy)),
                ]
            ),
        ]
    )


def two_local_pauli_lindblad_mpo(
    coeffs: tuple[float, ...],
    q1: int,
    q2: int,
    num_qubits: int,
    upper_ind_id: str = "k{}",
    lower_ind_id: str = "b{}",
) -> qtn.MatrixProductOperator:
    """MPO of a single two-qubit Pauli-Lindblad link in PTM form."""
    id = np.eye(4)
    diag_block = np.array([np.diag([1.0, 0.0, 0.0, 0.0]), np.diag([0.0, 1.0, 0.0, 0.0]),
                           np.diag([0.0, 0.0, 1.0, 0.0]), np.diag([0.0, 0.0, 0.0, 1.0])])
    spl_block = _two_local_pauli_lindblad_block(coeffs)

    tensors_data: list[np.ndarray] = []
    if q1 < q2:
        for q in range(q1):
            tensors_data.append(id.reshape(1, 4, 4) if q == 0 else id.reshape(1, 1, 4, 4))
        tensors_data.append(diag_block.reshape(4, 4, 4) if q1 == 0 else diag_block.reshape(1, 4, 4, 4))
        for _q in range(q1 + 1, q2):
            tensors_data.append(np.transpose(np.eye(16).reshape(4, 4, 4, 4), (0, 2, 1, 3)))
        tensors_data.append(spl_block.reshape(4, 4, 4) if q2 == num_qubits - 1 else spl_block.reshape(4, 1, 4, 4))
        for q in range(q2 + 1, num_qubits):
            tensors_data.append(id.reshape(1, 4, 4) if q == num_qubits - 1 else id.reshape(1, 1, 4, 4))
    if q1 > q2:
        for q in range(q2):
            tensors_data.append(id.reshape(1, 4, 4) if q == 0 else id.reshape(1, 1, 4, 4))
        tensors_data.append(spl_block.reshape(4, 4, 4) if q2 == 0 else spl_block.reshape(1, 4, 4, 4))
        for _q in range(q2 + 1, q1):
            tensors_data.append(np.transpose(np.eye(16).reshape(4, 4, 4, 4), (0, 2, 1, 3)))
        tensors_data.append(diag_block.reshape(4, 4, 4) if q1 == num_qubits - 1 else diag_block.reshape(4, 1, 4, 4))
        for q in range(q1 + 1, num_qubits):
            tensors_data.append(id.reshape(1, 4, 4) if q == num_qubits - 1 else id.reshape(1, 1, 4, 4))

    return qtn.tensor_1d.MatrixProductOperator(
        tensors_data,
        shape="lrud",
        site_tag_id="I{}",
        tags=None,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
    )


def sparse_pauli_lindblad_mpo_from_file(filename: str, num_qubits: int, max_qubits: int) -> qtn.MatrixProductOperator:
    """Build the MPO of a sparse Pauli-Lindblad noise model from a coefficient file."""
    single_coeffs, two_coeffs = parse_spl_coefficients(filename, num_qubits, max_qubits)

    tensors_data = []
    for q, (x, y, z) in enumerate(single_coeffs):
        local_ptm = np.diag([1.0, np.exp(-2.0 * y - 2.0 * z), np.exp(-2.0 * x - 2.0 * z), np.exp(-2.0 * x - 2.0 * y)])
        if q == 0 or q == num_qubits - 1:
            tensors_data.append(local_ptm.reshape(1, 4, 4))
        else:
            tensors_data.append(local_ptm.reshape(1, 1, 4, 4))

    mpo_noise = qtn.tensor_1d.MatrixProductOperator(
        tensors_data, shape="lrud", site_tag_id="I{}", tags=None, upper_ind_id="k{}", lower_ind_id="b{}"
    )

    for q_link, coeffs in enumerate(two_coeffs):
        mpo_noise = mpo_noise.apply(
            two_local_pauli_lindblad_mpo(coeffs, q_link, q_link + 1, num_qubits),
            compress=False,
        )

    return mpo_noise
