"""MPO construction helpers, Qiskit circuit→MPO conversion, and inversion routines.

Two parallel function families live here:

* **Unitary-basis builders** (`single_qubit_unitary_mpo`, `cu_to_mpo`, `qc_to_mpo`)
  construct rank-3 / rank-4 tensors that act on 2-D physical sites.
* **PTM-basis builders** (`identity_mpo_ptm`, `cnot_to_mpo_ptm`,
  `circuit_to_mpo_ptm`, `unitary_to_ptm`) act on 4-D physical sites (the Pauli
  Transfer Matrix representation used throughout the QPT pipeline).
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import qiskit as qk
import quimb
import quimb.tensor as qtn
import scipy as scp
from qiskit.quantum_info import Operator


def single_qubit_unitary_mpo(num_qubits: int, target_qubit: int, unitary_mat: np.ndarray) -> qtn.MatrixProductOperator:
    """Build a 2-D MPO acting as a single-qubit unitary on one site of a register.

    Args:
        num_qubits: total number of qubits.
        target_qubit: site index where the unitary acts.
        unitary_mat: 2×2 unitary matrix.
    """
    mpo = qtn.MPO_identity(num_qubits)

    if target_qubit == 0 or target_qubit == num_qubits - 1:
        unitary_mat = unitary_mat.reshape(1, 2, 2)
    else:
        unitary_mat = unitary_mat.reshape(1, 1, 2, 2)
    mpo[target_qubit].modify(data=unitary_mat)
    return mpo


def two_qubit_operation_mpo(num_qubits: int, target_qubits, local_operators: list) -> qtn.MatrixProductOperator:
    """Build an MPO for a generic two-qubit interaction of the form ``A_i ⊗ C_j + B_i ⊗ D_j``.

    For a CNOT with control on site i and target on site j the local operators
    are ``A = |0X0|, B = |1X1|, C = Id, D = X``. Works for both orderings
    ``i < j`` and ``i > j``.

    Args:
        num_qubits: total number of qubits.
        target_qubits: pair ``(q0, q1)`` of distinct site indices.
        local_operators: nested list specifying the two-term decomposition.
    """
    q0, q1 = target_qubits
    if q0 == q1:
        raise ValueError(f"Control and target qubit cannot be the same, got {q0} and {q1}!")

    qubit_ind = dict(zip((q0, q1), (0, 1), strict=True))

    id = np.eye(2)
    ops_list = []
    min_qub, max_qub = min(q0, q1), max(q0, q1)
    for site in range(num_qubits):
        if site == min_qub:
            if site == 0 or site == (num_qubits - 1):
                op = local_operators[qubit_ind[min_qub]]
            else:
                op = [local_operators[qubit_ind[min_qub]]]
        elif site < min_qub or site > max_qub:
            if site == 0 or site == (num_qubits - 1):
                op = id.reshape(1, 2, 2)
            else:
                op = id.reshape(1, 1, 2, 2)
        elif site == max_qub:
            if site == 0 or site == (num_qubits - 1):
                op = local_operators[qubit_ind[max_qub]]
            else:
                op = [o.reshape(1, 2, 2) for o in local_operators[qubit_ind[max_qub]]]
        else:
            op = [[id, 0 * id], [0 * id, id]]

        ops_list.append(np.array(op))

    return qtn.MatrixProductOperator(ops_list)


def cu_to_mpo(num_qubits: int, control_qubit: int, target_qubit: int, u: np.ndarray = None) -> qtn.MatrixProductOperator:
    """Build a controlled-unitary MPO with given control and target qubits."""
    if u is None:
        u = np.eye(2)
    proj_0 = np.array([[1, 0], [0, 0]])
    proj_1 = np.array([[0, 0], [0, 1]])
    id = np.eye(2)

    local_operators = [[proj_0, proj_1], [id, u]]
    return two_qubit_operation_mpo(num_qubits, (control_qubit, target_qubit), local_operators)


# CNOT and CZ unitary MPOs (2-D physical dimension).
cnot_to_mpo = partial(cu_to_mpo, u=np.array([[0, 1], [1, 0]]))
cz_to_mpo = partial(cu_to_mpo, u=np.array([[1, 0], [0, -1]]))


def ecr_to_mpo(num_qubits: int, control_qubit: int, target_qubit: int) -> qtn.MatrixProductOperator:
    """Build an Echo-Cross-Resonance MPO (used by IBM hardware)."""
    local_operators = [
        [
            1.0 / np.sqrt(2) * np.array([[0.0, 1.0], [1.0, 0.0]]),
            -1.0 / np.sqrt(2) * np.array([[0.0, -1j], [1j, 0.0]]),
        ],
        [np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])],
    ]
    return two_qubit_operation_mpo(num_qubits, (control_qubit, target_qubit), local_operators)


def qiskit_ordered_mpo(self: qtn.MatrixProductOperator, inplace: bool = False) -> qtn.MatrixProductOperator:
    """Reorder MPO indices last-to-first to match Qiskit's little-endian ordering."""
    tn = self if inplace else self.copy()
    num_qubits = tn.L

    reverse_output_inds = [*reversed(sorted(tn.outer_inds())[:num_qubits])] + [
        *reversed(sorted(tn.outer_inds())[num_qubits:])
    ]
    tn.reindex(dict(zip(sorted(tn.outer_inds()), reverse_output_inds, strict=True)), inplace=True)
    tn.retag(dict(zip(tn.site_tags, reversed(tn.site_tags), strict=False)), inplace=True)

    return tn


def qc_to_mpo(
    qc: qk.QuantumCircuit, check_with_qiskit: bool = False, compress: bool = False
) -> qtn.MatrixProductOperator:
    r"""Build a unitary MPO from a Qiskit ``QuantumCircuit``.

    The resulting MPO encodes ``U = \sum_{kb} U_{kb} |k\rangle\langle b|``
    with upper indices ``k`` and lower indices ``b``.

    Args:
        qc: input Qiskit circuit. Supported gates: any single-qubit gate, plus
            ``cx``, ``cz``, ``ecr`` two-qubit gates.
        check_with_qiskit: if True (or fewer than 9 qubits), verify the
            resulting MPO matches the dense unitary computed by Qiskit.
        compress: if True, compress the MPO after each gate application.
    """
    num_qubits = qc.num_qubits

    u_mpo = qtn.MPO_identity(num_qubits)
    for inst in qc._data:
        op = inst.operation
        target_qubits = [q._index for q in inst.qubits]

        if len(target_qubits) == 1:
            q_idx = target_qubits[0]
            u_mat = op.to_matrix()
            gate_mpo = single_qubit_unitary_mpo(num_qubits, q_idx, u_mat)

        elif len(target_qubits) == 2:
            control_q, target_q = target_qubits
            if op.name == "cx":
                gate_mpo = cnot_to_mpo(num_qubits, control_q, target_q)
            elif op.name == "cz":
                gate_mpo = cz_to_mpo(num_qubits, control_q, target_q)
            elif op.name == "ecr":
                gate_mpo = ecr_to_mpo(num_qubits, control_q, target_q)
            else:
                raise ValueError(f"Two-qubit operation {op.name} not implemented yet.")

        else:
            raise ValueError(f"Gate {op.name} acts on qubits {target_qubits}; not implemented yet.")

        u_mpo = gate_mpo.apply(u_mpo, compress=compress)

    u_mpo.mangle_inner_()

    if num_qubits < 9 or check_with_qiskit:
        uni_from_mpo = qiskit_ordered_mpo(u_mpo).to_dense()
        uni_from_qiskit = Operator(qc).data
        frob_norm = np.linalg.norm(uni_from_mpo - uni_from_qiskit)
        if frob_norm > 1e-12:
            raise ValueError(
                f"Unitary from MPO does not match Qiskit reference: ‖ U_mpo - U_qiskit ‖ = {frob_norm}"
            )

    return u_mpo


def dagger_mpo(self: qtn.MatrixProductOperator) -> qtn.MatrixProductOperator:
    """Return an MPO whose tensors are the transposed complex conjugate of the input."""
    tn = self.copy()
    data = []
    for t in tn:
        data.append(np.conj(np.swapaxes(t.data, -1, -2)))
    return qtn.MatrixProductOperator(data)


def evolve_obser_mpo(
    obs: qtn.MatrixProductOperator, u: qtn.MatrixProductOperator, compress: bool = False
) -> qtn.MatrixProductOperator:
    r"""Heisenberg evolution of an observable: ``O' = U^\dagger O U``."""
    u_dagger = dagger_mpo(u)
    return u_dagger.apply(obs, compress=compress).apply(u, compress=compress)


def evolve_state_mpo(
    rho: qtn.MatrixProductOperator, u: qtn.MatrixProductOperator, compress: bool = False
) -> qtn.MatrixProductOperator:
    r"""Schrödinger evolution of a state: ``rho' = U rho U^\dagger``."""
    u_dagger = dagger_mpo(u)
    return u.apply(rho, compress=compress).apply(u_dagger, compress=compress)


# ---------------------------------------------------------------------------
# PTM-basis MPO builders (moved from sabrina/tn_simulate_qpt.py — these act on
# 4-D physical sites, the Pauli Transfer Matrix representation).
# ---------------------------------------------------------------------------


def identity_mpo_ptm(num_qubits: int, tags=None, inds=("i", "j")) -> qtn.MatrixProductOperator:
    """Identity MPO on ``num_qubits`` qubits with 4-D PTM physical legs."""
    id = np.eye(4)
    tensors_data = []
    for q in range(num_qubits):
        if q == 0 or q == num_qubits - 1:
            tensors_data.append(id.reshape(1, 4, 4))
        else:
            tensors_data.append(id.reshape(1, 1, 4, 4))

    return qtn.tensor_1d.MatrixProductOperator(
        tensors_data,
        shape="lrud",
        site_tag_id="I{}",
        tags=tags,
        upper_ind_id=inds[1] + "_{}",
        lower_ind_id=inds[0] + "_{}",
    )


def unitary_to_ptm(u1: np.ndarray) -> np.ndarray:
    """Compute the 4×4 PTM of a single-qubit unitary."""
    ptm = np.eye(4)
    pauli_matrices = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ]

    for i in range(1, 4):
        for j in range(1, 4):
            ptm[i, j] = 0.5 * np.real(np.trace(np.dot(pauli_matrices[i], np.dot(u1, np.dot(pauli_matrices[j], np.conjugate(u1.T))))))
    return ptm


def cnot_to_mpo_ptm(q1: int, q2: int, num_qubits: int, tags=None) -> qtn.MatrixProductOperator:
    """CNOT MPO in PTM form (4-D physical legs), control on q1, target on q2."""

    def lambda_cnot(q, mu, mup):
        if (q == 0) and (mu == 0) and (mup == 0):
            return np.array([[1.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        if (q == 0) and (mu == 1) and (mup == 1):
            return np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        if (q == 0) and (mu == 1) and (mup == 0):
            return np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        if (q == 0) and (mu == 0) and (mup == 1):
            return np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 0]])
        if (q == 1) and (mu == 0) and (mup == 0):
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        if (q == 1) and (mu == 1) and (mup == 1):
            return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        if (q == 1) and (mu == 1) and (mup == 0):
            return np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        if (q == 1) and (mu == 0) and (mup == 1):
            return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        raise ValueError(f"Invalid (q, mu, mup) tuple: ({q}, {mu}, {mup}).")

    id = np.eye(4)
    tensors_data: list[np.ndarray] = []
    if q1 < q2:
        for q in range(q1):
            tensors_data.append(id.reshape(1, 4, 4) if q == 0 else id.reshape(1, 1, 4, 4))
        block = np.array([lambda_cnot(0, 0, 0), lambda_cnot(0, 0, 1), lambda_cnot(0, 1, 0), lambda_cnot(0, 1, 1)])
        tensors_data.append(block.reshape(4, 4, 4) if q1 == 0 else block.reshape(1, 4, 4, 4))
        for _q in range(q1 + 1, q2):
            tensors_data.append(np.transpose(np.eye(16).reshape(4, 4, 4, 4), (0, 2, 1, 3)))
        block = np.array([lambda_cnot(1, 0, 0), lambda_cnot(1, 0, 1), lambda_cnot(1, 1, 0), lambda_cnot(1, 1, 1)])
        tensors_data.append(block.reshape(4, 4, 4) if q2 == num_qubits - 1 else block.reshape(4, 1, 4, 4))
        for q in range(q2 + 1, num_qubits):
            tensors_data.append(id.reshape(1, 4, 4) if q == num_qubits - 1 else id.reshape(1, 1, 4, 4))
    if q1 > q2:
        for q in range(q2):
            tensors_data.append(id.reshape(1, 4, 4) if q == 0 else id.reshape(1, 1, 4, 4))
        block = np.array([lambda_cnot(1, 0, 0), lambda_cnot(1, 0, 1), lambda_cnot(1, 1, 0), lambda_cnot(1, 1, 1)])
        tensors_data.append(block.reshape(4, 4, 4) if q2 == 0 else block.reshape(1, 4, 4, 4))
        for _q in range(q2 + 1, q1):
            tensors_data.append(np.transpose(np.eye(16).reshape(4, 4, 4, 4), (0, 2, 1, 3)))
        block = np.array([lambda_cnot(0, 0, 0), lambda_cnot(0, 0, 1), lambda_cnot(0, 1, 0), lambda_cnot(0, 1, 1)])
        tensors_data.append(block.reshape(4, 4, 4) if q1 == num_qubits - 1 else block.reshape(4, 1, 4, 4))
        for q in range(q1 + 1, num_qubits):
            tensors_data.append(id.reshape(1, 4, 4) if q == num_qubits - 1 else id.reshape(1, 1, 4, 4))

    return qtn.tensor_1d.MatrixProductOperator(
        tensors_data,
        shape="lrud",
        site_tag_id="I{}",
        tags=tags,
        upper_ind_id="j_{}",
        lower_ind_id="i_{}",
    )


def circuit_to_mpo_ptm(qc: qk.QuantumCircuit, tags=("LAYER",), verbose: bool = False) -> qtn.MatrixProductOperator:
    """Build a PTM-form MPO from a Qiskit circuit (only single-qubit gates + CX supported)."""
    if verbose:
        print("---MPO Construction---")  # noqa: T201

    num_qubits = qc.width()
    num_gates = len(qc.data)
    circuit_mpo = identity_mpo_ptm(num_qubits, tags=list(tags))

    for gate_counter in range(num_gates):
        if verbose:
            print(f"Gate {gate_counter + 1}/{num_gates}", end="\r")  # noqa: T201

        gate = qc.data[gate_counter]
        if gate[0].name != "cx":
            q = int(gate[1][0]._index)
            u1 = gate[0].to_matrix()
            tensors_data = [np.eye(4)] * num_qubits
            tensors_data[q] = unitary_to_ptm(u1)
            u1_mpo = qtn.MPO_product_operator(tensors_data)
            circuit_mpo = u1_mpo.apply(circuit_mpo, compress=False)
        else:
            q1 = int(gate[1][0]._index)
            q2 = int(gate[1][1]._index)
            circuit_mpo = cnot_to_mpo_ptm(q1, q2, num_qubits).apply(circuit_mpo, compress=False)

    if verbose:
        print("\n---END---")  # noqa: T201

    return circuit_mpo


# ---------------------------------------------------------------------------
# MPO inversion / variational fitting utilities.
# ---------------------------------------------------------------------------


def variational_inversion_mpo(
    mpo: qtn.MatrixProductOperator,
    warmstart_mpo: qtn.MatrixProductOperator = None,
    virtual_bd: int = 4,
    optimizer: str = "L-BFGS-B",
    steps: int = 10_000,
    tol: float = 1e-12,
    jit_fn: bool = True,
    optimisation_method: str = "global",
    num_sweeps: int = 1,
    local_tags: list[Any] = None,
    verbose: bool = True,
) -> qtn.MatrixProductOperator:
    r"""Variationally find the inverse of an MPO by minimising ``‖ Λ(θ) Λ_targ − 𝟙 ‖_F``.

    Optimisation is performed via quimb's autodiff machinery (JAX backend).
    Mode is either ``"global"`` (all parameters at once) or ``"local"``
    (DMRG-style sweep over sites).
    """

    def norm_fn(mpo):
        return mpo

    target_mpo = mpo.copy()
    num_qubits = target_mpo.L
    phys_bd_size = target_mpo.ind_size(target_mpo.upper_ind_id.format(0))
    target_mpo.add_tag(["Target"])

    if warmstart_mpo is None:
        inv_mpo = qtn.MPO_rand(num_qubits, virtual_bd, phys_bd_size, tags=["Trainable"])
    else:
        inv_mpo = warmstart_mpo.copy()
        inv_mpo.add_tag(["Trainable"])

    local_tag_id = "O{}"
    for _, t in enumerate(inv_mpo.tensors):
        t.add_tag(local_tag_id.format(_))

    id_mpo = qtn.MPO_identity_like(mpo)

    if verbose:
        print("Before optimisation:")  # noqa: T201
        print(" > ‖ Λ(θ) - Λ_targ‖ = ", inv_mpo.distance(target_mpo))  # noqa: T201
        print(" > ‖ Λ(θ) - Id ‖ = ", inv_mpo.distance(id_mpo))  # noqa: T201
        print(" > ‖ Λ(θ) * Λ_targ  - Id ‖ = ", inv_mpo.apply(target_mpo).distance(id_mpo))  # noqa: T201

    tmp_index = "_tmp{}_"
    inv_mpo.reindex_lower_sites_(tmp_index)
    target_mpo.reindex_upper_sites_(tmp_index)
    tn = qtn.TensorNetwork([inv_mpo, target_mpo])

    if optimisation_method == "global":
        tn = qtn.tensor_network_fit_autodiff(
            tn,
            id_mpo,
            tags=["Trainable"],
            norm_fn=norm_fn,
            optimizer=optimizer,
            steps=steps,
            tol=tol,
            progbar=True,
            inplace=True,
            autodiff_backend="jax",
            jit_fn=jit_fn,
        )

    if optimisation_method == "local":
        if local_tags is None:
            local_tags = [local_tag_id.format(_) for _ in range(num_qubits)]

        for sweep_idx in range(num_sweeps):
            for t in local_tags:
                tn = qtn.tensor_network_fit_autodiff(
                    tn,
                    id_mpo,
                    tags=t,
                    norm_fn=norm_fn,
                    optimizer=optimizer,
                    steps=steps,
                    tol=tol,
                    progbar=True,
                    inplace=True,
                    autodiff_backend="jax",
                    jit_fn=jit_fn,
                )
            for t in local_tags[::-1][1:-1]:
                tn = qtn.tensor_network_fit_autodiff(
                    tn,
                    id_mpo,
                    tags=t,
                    norm_fn=norm_fn,
                    optimizer=optimizer,
                    steps=steps,
                    tol=tol,
                    progbar=True,
                    inplace=True,
                    autodiff_backend="jax",
                    jit_fn=jit_fn,
                )

    opt_mpo = tn.select(["Trainable"]).view_as(
        qtn.MatrixProductOperator,
        L=num_qubits,
        site_tag_id="I{}",
        upper_ind_id=inv_mpo.upper_ind_id,
        lower_ind_id=inv_mpo.lower_ind_id,
        cyclic=False,
    )
    opt_mpo.reindex_({tmp_index.format(_): inv_mpo.lower_ind_id.format(_) for _ in range(num_qubits)})
    opt_mpo.drop_tags(["Trainable"])
    return opt_mpo


def invert_mpo(
    mpo: qtn.MatrixProductOperator,
    bond_dimension: int,
    num_sweeps: int,
    canonize_while_sweeping: bool = True,
    seed: int = None,
    dtype: type = float,
    imag_tol: float = 1e-12,
):
    r"""Iteratively invert an MPO via DMRG-style local linear solves.

    Returns the inverse MPO ``B`` such that ``B * A ≈ 𝟙`` and the final
    Frobenius residual.
    """

    def loss_fn(mpo_one, mpo_two):
        mpo = mpo_one.copy()
        mpo.upper_ind_id = "_u_tmp_{}"
        mpo.lower_ind_id = "_l_tmp_{}"
        return (mpo.apply(mpo_two)).distance(mpo_two.identity())

    def add_tags_and_rename_inner_inds(m, local_tags_id, inner_inds_id):
        for _, t in enumerate(m.tensors):
            t.add_tag(local_tags_id.format(_))
        m.reindex_(
            {old: new for old, new in zip(m.inner_inds(), (inner_inds_id.format(s + 1) for s in range(num_sites)), strict=False)}
        )

    num_sites = mpo.L
    phys_dim = mpo.phys_dim()

    mpo_A = qtn.MatrixProductOperator(mpo.arrays, shape="lrud", upper_ind_id="k{}", lower_ind_id="j{}", site_tag_id="I{}")
    add_tags_and_rename_inner_inds(mpo_A, local_tags_id="A{}", inner_inds_id="a_bond{}")

    mpo_Adag = qtn.MatrixProductOperator(
        (mpo_A.H).arrays, shape="lrdu", upper_ind_id="l{}", lower_ind_id="k{}", site_tag_id="I{}"
    )
    add_tags_and_rename_inner_inds(mpo_Adag, local_tags_id="Adag{}", inner_inds_id="adag_bond{}")

    if seed is not None:
        quimb.gen.rand.seed_rand(seed)
    mpo_B = qtn.MPO_rand(
        num_sites,
        bond_dim=bond_dimension,
        phys_dim=phys_dim,
        upper_ind_id="j{}",
        lower_ind_id="i{}",
        site_tag_id="I{}",
        dtype=dtype,
    )
    add_tags_and_rename_inner_inds(mpo_B, local_tags_id="B{}", inner_inds_id="b_bond{}")
    if canonize_while_sweeping:
        mpo_B.right_canonize()

    mpo_Bdag = qtn.MatrixProductOperator(
        (mpo_B.H).arrays, shape="lrdu", upper_ind_id="m{}", lower_ind_id="l{}", site_tag_id="I{}"
    )
    add_tags_and_rename_inner_inds(mpo_Bdag, local_tags_id="Bdag{}", inner_inds_id="bdag_bond{}")

    id_mpo_tn1 = qtn.MPO_identity(num_sites, phys_dim, upper_ind_id="m{}", lower_ind_id="k{}")
    id_mpo_tn2 = qtn.MPO_identity(num_sites, phys_dim, upper_ind_id="i{}", lower_ind_id="m{}")

    print(f"Initial distance between B and A^-1: ‖ B * A - Id ‖_Frob = {loss_fn(mpo_B, mpo_A)}")  # noqa: T201

    for sweep in range(num_sweeps):
        sweep_sites = range(num_sites) if sweep % 2 == 0 else range(num_sites - 1)[::-1]

        for site in sweep_sites:
            tn1 = qtn.TensorNetwork([mpo_Adag, mpo_Bdag, id_mpo_tn1])
            tn2 = qtn.TensorNetwork([mpo_B, mpo_A, mpo_Adag, mpo_Bdag, id_mpo_tn2])

            env_tensor = tn2.select(tags=[f"B{site}", f"Bdag{site}"], which="!any").contract(optimize="hyper")
            env_tensor = env_tensor.trace(mpo_B.lower_ind_id.format(site), mpo_Bdag.upper_ind_id.format(site)) / phys_dim

            left_env_inds = [bd for bd in tn2[f"Bdag{site}"].inds if bd not in (mpo_Bdag.upper_ind_id.format(site))]
            right_env_inds = [bd for bd in tn2[f"B{site}"].inds if bd not in (mpo_B.lower_ind_id.format(site))]
            env_matrix = env_tensor.to_dense(left_env_inds, right_env_inds)

            env_vector = tn1.select(tags=[f"Bdag{site}"], which="!any").contract(optimize="hyper").to_dense(tn2[f"Bdag{site}"].inds)

            b_opt = np.kron(scp.linalg.inv(env_matrix), np.eye(phys_dim)) @ env_vector
            if dtype is float:
                b_opt = np.real_if_close(b_opt, tol=imag_tol)

            mpo_B.tensors[site].modify(data=b_opt.reshape(mpo_B.tensors[site].data.shape))

            if canonize_while_sweeping:
                if site == 0:
                    if sweep % 2 == 0:
                        mpo_B.shift_orthogonality_center(site, site + 1)
                elif site == num_sites - 1:
                    if sweep != (num_sweeps - 1):
                        mpo_B.shift_orthogonality_center(site, site - 1)
                else:
                    if sweep % 2 == 0:
                        mpo_B.shift_orthogonality_center(site, site + 1)
                    else:
                        mpo_B.shift_orthogonality_center(site, site - 1)

            mpo_Bdag = qtn.MatrixProductOperator(
                (mpo_B.H).arrays, shape="lrdu", upper_ind_id="m{}", lower_ind_id="l{}", site_tag_id="I{}"
            )
            add_tags_and_rename_inner_inds(mpo_Bdag, local_tags_id="Bdag{}", inner_inds_id="bdag_bond{}")

            Acopy = qtn.MatrixProductOperator(mpo_A.arrays, shape="lrud")
            Bcopy = qtn.MatrixProductOperator(mpo_B.arrays, shape="lrud")
            val = (Bcopy.apply(Acopy) - Acopy.identity()).norm()
            print(f"Iteration {sweep + 1}, qubit {site}, ||A*B-I||_F = {val}")  # noqa: T201

    return mpo_B, loss_fn(mpo_B, mpo_A)


def mpo_to_params_file(mpo: qtn.MatrixProductOperator, file_name: str, save_key: str = None) -> None:
    """Serialise an MPO's tensors to a JSON file (creating or appending under ``save_key``)."""
    params_dict = {k: v.astype(str).tolist() for k, v in mpo.get_params().items()}
    path = Path(file_name)

    if not path.is_file():
        with path.open("w") as f:
            json.dump(params_dict, f, indent=4)
        return

    with path.open("r") as f:
        existing_data = json.load(f)

    if save_key is not None:
        existing_data[save_key] = params_dict

    with path.open("w") as f:
        json.dump(existing_data, f, indent=4)


def mpo_from_params_file(file_name: str, load_key: str = None) -> qtn.MatrixProductOperator:
    """Load an MPO previously saved with `mpo_to_params_file`."""
    with Path(file_name).open("r") as f:
        params = json.load(f)

    if load_key is not None:
        params = params[load_key]

    params = {n: np.array(v, dtype=complex) for n, (_k, v) in enumerate(params.items())}

    num_qubits = len(params)
    bond_dim, phys_dim, _ = params[list(params.keys())[0]].shape

    mpo = qtn.MPO_rand(num_qubits, bond_dim, phys_dim)
    mpo.set_params(params)
    return mpo
