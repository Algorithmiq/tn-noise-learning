"""Ideal circuit as a tensor network that conjugates input states, ``ρ → U ρ U†``.

Each two-qubit gate is split by an operator-Schmidt decomposition into two site tensors joined by a bond, so the
construction works on any qubit connectivity without compression. Raw states enter on legs ``p{j}`` (ket) and
``q{j}`` (bra) and leave on the LPDO input legs ``k{j}`` and ``b{j}``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from qiskit.quantum_info import Operator

from .lpdo import input_ind, site_tag

if TYPE_CHECKING:
    from jax import Array
    from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

SCHMIDT_TOL = 1e-12
LARGE_TENSOR_SIZE = 10_000_000
BATCH_IND = "exp"


def raw_ind(site: int, *, dagger: bool = False) -> str:
    """Leg of a raw (not yet evolved) state: ket side ``p{j}`` or bra side ``q{j}``."""
    return f"{'q' if dagger else 'p'}{site}"


def _schmidt_pair(gate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 gate on qubits ``(a, b)`` (little-endian) into ``A[α, out_a, in_a]`` and ``B[α, out_b, in_b]``."""
    matrix = gate.reshape(2, 2, 2, 2).transpose(1, 3, 0, 2).reshape(4, 4)
    u, s, vh = np.linalg.svd(matrix)
    rank = int(np.sum(s > SCHMIDT_TOL))
    a = (u[:, :rank] * np.sqrt(s[:rank])).T.reshape(rank, 2, 2)
    b = (vh[:rank] * np.sqrt(s[:rank])[:, None]).reshape(rank, 2, 2)
    return a, b


def circuit_network(circuit: QuantumCircuit) -> qtn.TensorNetwork:
    """One tensor per site mapping ``(p_j, q_j)`` to ``(k_j, b_j)`` such that the whole network is ``U · U†``.

    Only one- and two-qubit unitary gates are supported; barriers are skipped.
    """
    num_sites = circuit.num_qubits
    identity = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
    tn = qtn.TensorNetwork(
        [
            qtn.Tensor(
                identity,
                inds=(raw_ind(j), raw_ind(j, dagger=True), input_ind(j), input_ind(j, dagger=True)),
                tags=site_tag(j),
            )
            for j in range(num_sites)
        ]
    )
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name == "barrier":
            continue
        sites = [circuit.find_bit(q).index for q in instruction.qubits]
        if len(sites) > 2:
            raise ValueError(
                f"Gate {operation.name} acts on {len(sites)} qubits; only one- and two-qubit gates are supported."
            )
        try:
            gate = np.asarray(Operator(operation).data, dtype=complex)
        except Exception as error:
            raise ValueError(f"Operation {operation.name} is not a unitary gate.") from error
        pieces = [(sites[0], gate[None])] if len(sites) == 1 else list(zip(sites, _schmidt_pair(gate), strict=True))
        bond_ket, bond_bra = qtn.rand_uuid("s"), qtn.rand_uuid("s")
        for site, piece in pieces:
            old_ket, old_bra = qtn.rand_uuid("s"), qtn.rand_uuid("s")
            tn.reindex_({input_ind(site): old_ket, input_ind(site, dagger=True): old_bra})
            tn.add_tensor(qtn.Tensor(piece, inds=(bond_ket, input_ind(site), old_ket), tags=site_tag(site)))
            tn.add_tensor(
                qtn.Tensor(piece.conj(), inds=(bond_bra, input_ind(site, dagger=True), old_bra), tags=site_tag(site))
            )
    for j in range(num_sites):
        tn.contract_tags_(site_tag(j))
    tn.fuse_multibonds_()
    tn.squeeze_()
    largest = max(t.size for t in tn)
    if largest > LARGE_TENSOR_SIZE:
        logger.warning(
            "Circuit network has a site tensor with %d elements; the circuit is deep for an uncompressed decomposition",
            largest,
        )
    return tn


def product_state_network(states: np.ndarray | Array, *, raw: bool = True) -> qtn.TensorNetwork:
    """Batched product states ``states[j, exp]`` of shape ``(num_sites, batch, 2, 2)`` as site tensors.

    Each tensor has legs ``(exp, ket_j, bra_j)``: ``p{j}, q{j}`` with ``raw=True`` (to be evolved by a circuit),
    otherwise ``k{j}, b{j}``.
    """
    ket, bra = (raw_ind, raw_ind) if raw else (input_ind, input_ind)
    return qtn.TensorNetwork(
        [qtn.Tensor(s, inds=(BATCH_IND, ket(j), bra(j, dagger=True)), tags=site_tag(j)) for j, s in enumerate(states)]
    )


def apply_circuit(states_tn: qtn.TensorNetwork, circuit: QuantumCircuit) -> qtn.TensorNetwork:
    """Evolve a raw state network (legs ``p, q``) to ``U ρ U†`` with legs ``k, b``, one tensor per site."""
    tn = qtn.TensorNetwork([states_tn, circuit_network(circuit)], check_collisions=False)
    for j in range(circuit.num_qubits):
        tn.contract_tags_(site_tag(j))
    return tn
