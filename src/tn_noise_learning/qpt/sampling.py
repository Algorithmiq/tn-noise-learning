"""State/POVM selectors and local sampling strategies for QPT experiments."""

from __future__ import annotations

import itertools

import numpy as np
import quimb.tensor as qtn

from .states import labels_to_tensors


def state_selector(alpha, tags=("IN",), ind: str = "a", state_space: int = 6) -> qtn.MatrixProductState:
    r"""Product MPS that picks a specific input state from an overcomplete set.

    Each leg has dimension ``state_space`` and is a one-hot vector picking
    one of the candidate states::

        O--O--...--O--O
        |  |       |  |
        d  d       d  d        (d = state_space)
    """
    alpha = np.array(alpha)
    alpha_one_hot = labels_to_tensors(alpha, state_space=state_space)
    return qtn.MPS_product_state(alpha_one_hot, tags=list(tags), site_ind_id=ind + "_{}")


def povm_selector(beta, tags=("OUT",), ind: str = "b", state_space: int = 6) -> qtn.MatrixProductState:
    """Product MPS that picks a specific POVM effect from a set (see `state_selector`)."""
    beta = np.array(beta)
    beta_one_hot = labels_to_tensors(beta, state_space=state_space)
    return qtn.MPS_product_state(beta_one_hot, tags=["OUT"], site_ind_id=ind + "_{}")


def flatten_list(data):
    """Concatenate one level of nested lists."""
    flat_list = []
    for row in data:
        flat_list.extend(row)
    return flat_list


def local_preps(
    num_qubits: int, locality: int = 3, num_preps: int | None = None, num_states: int = 6
) -> list[list[int]]:
    """Enumerate all length-``locality`` preparation tuples tiled across ``num_qubits``.

    Args:
        num_qubits: register size.
        locality: number of contiguous qubits over which preparations vary.
        num_preps: if given, repeat the list to reach approximately this many entries.
        num_states: cardinality of the single-qubit state set
            (6 for OC Pauli, 4 for SIC).
    """
    states_label = list(range(num_states))
    local_full_preps = np.array(list(itertools.product(states_label, repeat=locality)))
    preps_list = np.tile(local_full_preps, int(np.ceil(num_qubits / locality)))[:, :num_qubits]

    if isinstance(num_preps, int):
        preps_list = [preps_list] * int(np.ceil(num_preps / len(preps_list)))
        preps_list = flatten_list(preps_list)

    return list(map(list, preps_list))


def local_meas(num_qubits: int, locality: int = 3, num_preps: int | None = None, num_basis: int = 3) -> list[list[int]]:
    """Enumerate all length-``locality`` measurement-basis tuples tiled across ``num_qubits``."""
    basis_labels = list(range(num_basis))
    local_full_meas = np.array(list(itertools.product(basis_labels, repeat=locality)))
    meas_list = np.tile(local_full_meas, int(np.ceil(num_qubits / locality)))[:, :num_qubits]

    if isinstance(num_preps, int):
        meas_list = [meas_list] * int(np.ceil(num_preps / len(meas_list)))
        meas_list = flatten_list(meas_list)

    return list(map(list, meas_list))


def local_preps_and_meas(
    num_qubits: int, locality: int = 2, num_shots: int = 1000, num_states: int = 6, random_pick: bool = False
):
    """Cartesian product of local preparations and measurement bases, repeated to ``num_shots`` entries."""
    input_states = local_preps(num_qubits, locality=locality, num_states=num_states)
    basis_list = local_meas(num_qubits, locality=locality)
    print(f"--> Selected {len(input_states)} input states and {len(basis_list)} measurement basis.\n")  # noqa: T201

    preps_and_meas = list(itertools.product(input_states, basis_list))

    if num_shots > len(preps_and_meas):
        preps_and_meas = flatten_list([preps_and_meas] * int(np.ceil(num_shots / len(preps_and_meas))))
        if random_pick:
            np.random.shuffle(preps_and_meas)
            preps_and_meas = preps_and_meas[:num_shots]

    return preps_and_meas
