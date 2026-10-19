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

from __future__ import annotations

from functools import reduce

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from tn_noise_learning.circuits import apply_circuit, circuit_network, product_state_network


def chain_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rzz(0.3, 1, 2)
    qc.rx(0.7, 2)
    qc.cx(2, 0)
    return qc


def square_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.cz(0, 1)
    qc.cz(2, 3)
    qc.rzz(0.5, 0, 3)
    qc.barrier()
    qc.sx(1)
    return qc


def dense_evolved(states: np.ndarray, qc: QuantumCircuit, shot: int) -> np.ndarray:
    rho = reduce(np.kron, [states[j, shot] for j in reversed(range(qc.num_qubits))])
    u = Operator(qc).data
    return u @ rho @ u.conj().T


@pytest.mark.parametrize("qc", [chain_circuit(), square_circuit()], ids=["chain3", "square22"])
def test_apply_circuit_matches_dense_operator(qc):
    rng = np.random.default_rng(0)
    states = rng.standard_normal((qc.num_qubits, 3, 2, 2)) + 1j * rng.standard_normal((qc.num_qubits, 3, 2, 2))
    evolved = apply_circuit(product_state_network(states), qc)
    n = qc.num_qubits
    kets = [f"k{j}" for j in reversed(range(n))]
    bras = [f"b{j}" for j in reversed(range(n))]
    for shot in range(3):
        dense = evolved.contract(output_inds=["exp", *kets, *bras]).data[shot].reshape(2**n, 2**n)
        np.testing.assert_allclose(dense, dense_evolved(states, qc, shot), atol=1e-12)


def test_cx_bond_dimension():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    tn = circuit_network(qc)
    (bond,) = tn.inner_inds()
    assert tn.ind_size(bond) == 4


def test_repeated_gates_fuse_into_one_bond():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)
    tn = circuit_network(qc)
    assert len(tn.inner_inds()) == 1
    assert tn.ind_size(next(iter(tn.inner_inds()))) == 16


def test_rejects_three_qubit_and_non_unitary():
    qc = QuantumCircuit(3, 1)
    qc.ccx(0, 1, 2)
    with pytest.raises(ValueError, match="only one- and two-qubit"):
        circuit_network(qc)
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with pytest.raises(ValueError, match="not a unitary"):
        circuit_network(qc)
