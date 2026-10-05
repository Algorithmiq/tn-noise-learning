"""Synthetic datasets from a known channel: exact outcome distributions sampled one setting at a time.

A setting is a pair ``(state_labels, basis_labels)`` with one entry per qubit. One exact contraction gives the joint
distribution over all ``2**num_qubits`` bit strings of that setting, from which all its shots are drawn with a single
multinomial draw. Effect labels follow the convention ``2 * basis + bit``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit

from .channels import pauli_eigenstates
from .circuits import apply_circuit, raw_ind
from .dataset import NoiseDataset
from .lpdo import LPDO, input_ind, output_ind, site_tag

MAX_SAMPLED_SITES = 20
Setting = tuple[Sequence[int], Sequence[int]]


def _check_bases(povms: np.ndarray) -> None:
    if povms.shape[1] % 2 or not np.allclose(povms.reshape(povms.shape[0], -1, 2, 2, 2).sum(axis=2), np.eye(2)):
        raise ValueError("povms must list, per qubit, consecutive pairs of effects that sum to the identity.")


def outcome_distribution(
    channel: LPDO,
    circuit: QuantumCircuit | None,
    state_labels: Sequence[int],
    basis_labels: Sequence[int],
    states: np.ndarray,
    povms: np.ndarray,
    optimize: str = "auto-hq",
) -> np.ndarray:
    """Joint probability of every bit string for one setting, shape ``(2,) * num_qubits`` with qubit 0 first."""
    n = channel.num_sites
    if n > MAX_SAMPLED_SITES:
        raise ValueError(f"Exact sampling enumerates 2**n outcomes; n must be at most {MAX_SAMPLED_SITES}, got {n}.")
    prep = qtn.TensorNetwork(
        [qtn.Tensor(states[j, state_labels[j]], inds=(raw_ind(j), raw_ind(j, dagger=True)), tags=site_tag(j)) for j in range(n)]
    )
    if circuit is None:
        prep.reindex_({raw_ind(j): input_ind(j) for j in range(n)} | {raw_ind(j, dagger=True): input_ind(j, dagger=True) for j in range(n)})
    else:
        prep = apply_circuit(prep, circuit)
    measurement = qtn.TensorNetwork(
        [
            qtn.Tensor(povms[j, 2 * b : 2 * b + 2], inds=(f"o{j}", output_ind(j, dagger=True), output_ind(j)), tags=site_tag(j))
            for j, b in enumerate(basis_labels)
        ]
    )
    tn = qtn.TensorNetwork([channel.network(), prep, measurement], check_collisions=False)
    probs = np.real(tn.contract(output_inds=[f"o{j}" for j in range(n)], optimize=optimize).data)
    probs = np.clip(probs, 0, None)
    return probs / probs.sum()


def sample_shots(
    channel: LPDO,
    circuit: QuantumCircuit | None = None,
    *,
    num_settings: int = 100,
    shots_per_setting: int = 100,
    seed: int | None = None,
    states: np.ndarray | None = None,
    povms: np.ndarray | None = None,
    settings: Sequence[Setting] | None = None,
) -> NoiseDataset:
    """Sample a dataset from ``channel`` applied after ``circuit``.

    Args:
        channel: ground-truth channel; stored in the dataset as ``ideal_channel``.
        circuit: ideal circuit applied before the channel (``None`` for the identity).
        num_settings: number of random ``(preparation, basis)`` settings when ``settings`` is not given.
        shots_per_setting: shots drawn per setting.
        seed: seed for settings, shots and the final shuffle.
        states: input states per qubit, shape ``(num_qubits, num_states, 2, 2)``; default Pauli eigenstates.
        povms: effects per qubit as consecutive basis pairs; default Pauli eigenstates.
        settings: explicit settings, overriding ``num_settings``.
    """
    rng = np.random.default_rng(seed)
    n = channel.num_sites
    states = pauli_eigenstates(n) if states is None else np.asarray(states, dtype=complex)
    povms = pauli_eigenstates(n) if povms is None else np.asarray(povms, dtype=complex)
    _check_bases(povms)
    if settings is None:
        settings = [(rng.integers(0, states.shape[1], n), rng.integers(0, povms.shape[1] // 2, n)) for _ in range(num_settings)]
    bits = np.array(np.unravel_index(np.arange(2**n), (2,) * n)).T
    inputs, outcomes = [], []
    for prep, basis in settings:
        prep, basis = np.asarray(prep), np.asarray(basis)
        counts = rng.multinomial(shots_per_setting, outcome_distribution(channel, circuit, prep, basis, states, povms).ravel())
        for flat in np.flatnonzero(counts):
            inputs.append(np.repeat(prep[None], counts[flat], axis=0))
            outcomes.append(np.repeat((2 * basis + bits[flat])[None], counts[flat], axis=0))
    order = rng.permutation(len(settings) * shots_per_setting)
    return NoiseDataset(np.concatenate(inputs)[order], np.concatenate(outcomes)[order], states, povms, circuit, ideal_channel=channel)
