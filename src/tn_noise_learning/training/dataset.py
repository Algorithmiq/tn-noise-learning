"""Dataset preprocessing: turn raw QPT JSON counts into batched tensor networks."""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from quimb.tensor import TensorNetwork
from tqdm import tqdm

from ..tn_core.mpo_utils import qc_to_mpo


def experiments_to_conditional(experiments: dict) -> dict:
    """Group experimental outcomes by their input state.

    For each unique input state (as a string label), returns the list of all
    measured POVM outcomes recorded for that state.
    """
    unique_inputs = [list(v.keys())[0] for v in experiments.values()]
    unique_inputs = np.unique(unique_inputs)

    out_given_in: dict[str, list[str]] = {}
    for rho in unique_inputs:
        povms_given_rho = []
        for vals in experiments.values():
            try:
                povms_given_rho.append(vals[rho])
            except KeyError:
                pass
        out_given_in[rho] = povms_given_rho
    return out_given_in


def experiments_to_tensors(experiments: dict) -> np.ndarray:
    """Rearrange experiment counts to a ``(num_shots, 2, num_qubits)`` integer array."""
    res = []
    for exp in experiments.values():
        alpha, beta = list(exp.items())[0]
        alpha = [int(s) for s in alpha]
        beta = [int(s) for s in beta]
        res.append(np.array([alpha, beta]))
    return np.array(res)


def unique_elements_list(data_list):
    """Stable de-duplication preserving first-occurrence order."""
    if not isinstance(data_list, list):
        data_list = data_list.tolist()
    unique_list: list = []
    for s in data_list:
        if s not in unique_list:
            unique_list.append(s)
    return unique_list


def flatten_list(data):
    """Concatenate one level of nested lists."""
    flat_list = []
    for row in data:
        flat_list.extend(row)
    return flat_list


def brickwall_marginals_inds(num_qubits: int, effect_support: int = 2, return_full_settings: bool = False):
    """Enumerate brickwall lightcone supports for state preparations and measurement effects."""
    local_systems_at_measurement = [list(range(n, n + effect_support)) for n in range(num_qubits - effect_support + 1)]

    inds = []
    for supported_qubs_at_measurement in local_systems_at_measurement:
        up_qubit_effect = min(supported_qubs_at_measurement)
        low_qubit_effect = max(supported_qubs_at_measurement)

        if not up_qubit_effect % 2:
            up_qubit_state = max(up_qubit_effect - 2, 0)
        else:
            up_qubit_state = max(up_qubit_effect - 1, 0)

        if not low_qubit_effect % 2:
            low_qubit_state = min(low_qubit_effect + 1, num_qubits - 1)
        else:
            low_qubit_state = min(low_qubit_effect + 2, num_qubits - 1)

        supported_qubs_at_preparation = list(range(up_qubit_state, low_qubit_state + 1))

        if return_full_settings:
            qub_inds = np.array([None] * num_qubits, dtype=object)
            full_state = qub_inds.copy()
            full_effect = qub_inds.copy()
            full_state[supported_qubs_at_preparation] = supported_qubs_at_preparation
            full_effect[supported_qubs_at_measurement] = supported_qubs_at_measurement
            inds.append([full_state.tolist(), full_effect.tolist()])
        else:
            inds.append([supported_qubs_at_preparation, supported_qubs_at_measurement])
    return inds


def lightcones_marginal_experiments(
    counts: np.ndarray, lightcone_structure: str | list = "brickwall", include_original_counts: bool = True
) -> np.ndarray:
    """Replicate the count array with each setting marginalised over a brickwall lightcone."""
    num_qubits = counts.shape[-1]

    if lightcone_structure == "brickwall":
        lightcone_inds = brickwall_marginals_inds(num_qubits)
    else:
        raise ValueError(f"{lightcone_structure} lightcone structure is not specified.")

    marginalised_counts = np.empty((len(lightcone_inds), len(counts), 2, num_qubits), dtype=object)
    marginalised_counts.fill("Id")

    for n, (state_support, effect_support) in tqdm(list(enumerate(lightcone_inds))):
        marginalised_counts[n, :, 0, state_support] = counts[:, 0, state_support].T
        marginalised_counts[n, :, 1, effect_support] = counts[:, 1, effect_support].T

    marginalised_counts = marginalised_counts.reshape(len(counts) * len(lightcone_inds), 2, num_qubits)

    if include_original_counts:
        marginalised_counts = np.vstack((counts, marginalised_counts))
    return marginalised_counts


def batch_experiments(experiments: np.ndarray, batch_size: int = 32, shuffle: bool = True, verbose: bool = False):
    """Split an experiment-count array into uniformly sized batches.

    Repeats some rows so each batch has exactly ``batch_size`` entries —
    introduces a small bias for the final batch but keeps JAX shapes static.
    """
    if shuffle:
        np.random.shuffle(experiments)

    num_batches = int(np.ceil(len(experiments) / batch_size))
    experiments = np.vstack((experiments, experiments[: batch_size * num_batches - len(experiments)]))

    batch_inds = np.array_split(range(len(experiments)), num_batches)
    batched_experiments = experiments[batch_inds]
    if verbose:
        print(f"Experiments array shape: {batched_experiments.shape} (num_batches, batch_size, 2, num_qubits)")  # noqa: T201

    return batched_experiments


def experiments_to_hyper_tn(experiments: list, states: list, povms: list):
    """Build the hyper-indexed state/POVM tensor networks for a single batch."""
    states = np.array(states, dtype=complex)
    states = [{"Id": np.eye(2, dtype=complex) / 2} | dict(enumerate(s)) for s in states]

    povms = np.array(povms, dtype=complex)
    povms = [{"Id": np.eye(2, dtype=complex)} | dict(enumerate(p)) for p in povms]

    in_exp = []
    out_exp = []
    for alpha, beta in experiments:
        in_state_single_exp = []
        out_povm_single_exp = []
        for num_qubit, (a, b) in enumerate(zip(alpha, beta, strict=True)):
            in_state_single_exp.append(states[num_qubit][a])
            out_povm_single_exp.append(povms[num_qubit][b])
        in_exp.append(in_state_single_exp)
        out_exp.append(out_povm_single_exp)

    in_exp = np.array(in_exp)
    out_exp = np.array(out_exp)
    in_exp = np.swapaxes(in_exp, 0, 1)
    out_exp = np.swapaxes(out_exp, 0, 1)

    exp_states = [qtn.Tensor(sq, inds=("exp", f"k{j}", f"b{j}"), tags=(f"I{j}", "IN")) for j, sq in enumerate(in_exp)]
    exp_povms = [
        qtn.Tensor(sq, inds=("exp", f"bn{j}", f"kn{j}"), tags=(f"I{j}", "OUT")) for j, sq in enumerate(out_exp)
    ]
    hyper_states_tn = qtn.TensorNetwork(exp_states)
    hyper_povm_tn = qtn.TensorNetwork(exp_povms)
    return hyper_states_tn, hyper_povm_tn


def evolve_hyper_tn(hyper_states_tn: TensorNetwork, u: QuantumCircuit | TensorNetwork):
    r"""Conjugate the hyper-state TN by an ideal unitary: ``S → U S U^\dagger``."""
    if isinstance(u, QuantumCircuit):
        num_qubits = u.num_qubits
        u_mpo = qc_to_mpo(u)
    else:
        raise TypeError(f"Unitary must be a Qiskit QuantumCircuit, got {type(u)}.")

    tmp_left = "_v{}_"
    tmp_right = "_u{}_"

    reindex_mpo = {f"b{n}": tmp_left.format(n) for n in range(num_qubits)}
    u_left = u_mpo.reindex(reindex_mpo)

    reindex_state = {f"k{n}": tmp_left.format(n) for n in range(num_qubits)} | {
        f"b{n}": tmp_right.format(n) for n in range(num_qubits)
    }
    s = hyper_states_tn.reindex(reindex_state)

    reindex_mpo_dagger = {f"b{n}": tmp_right.format(n) for n in range(num_qubits)} | {
        f"k{n}": f"b{n}" for n in range(num_qubits)
    }
    udag = (u_mpo.copy().H).reindex(reindex_mpo_dagger)

    tn = qtn.TensorNetwork([u_left, s, udag])

    for n in range(num_qubits):
        tn.contract_tags_(f"I{n}")
    tn.fuse_multibonds_()
    for t in tn.tensors:
        t.moveindex_("exp", 0)
    for n, t in enumerate(tn.tensors):
        t.moveindex_(f"k{n}", -2)
        t.moveindex_(f"b{n}", -1)

    return tn


def preprocess_experiments(
    experiments: dict, batch_size: int = 32, shuffle: bool = True, build_marginals: str | None = None
):
    """Tensorise + (optionally marginalise) + batch the experiment counts."""
    counts = experiments_to_tensors(experiments)

    if isinstance(build_marginals, str):
        print(f"> Generating marginals assuming {build_marginals} causal structure")  # noqa: T201
        counts = lightcones_marginal_experiments(counts, lightcone_structure=build_marginals)

    print(f"Shots = {counts.shape}")  # noqa: T201
    return batch_experiments(counts, batch_size=batch_size, shuffle=shuffle)


def dataset_generator(batched_experiments, states: list, povms: list, evolve_with_unitary=None):
    """Lazy generator yielding ``((state_params, state_skel), (povm_params, povm_skel))`` per batch."""
    for batch in batched_experiments:
        batch_tn_states, batch_tn_povms = experiments_to_hyper_tn(batch, states, povms)
        if evolve_with_unitary is not None:
            batch_tn_states = evolve_hyper_tn(batch_tn_states, evolve_with_unitary)
        yield qtn.pack(batch_tn_states), qtn.pack(batch_tn_povms)


def create_training_dataset(
    experiments: dict,
    states: list,
    povms: list,
    batch_size: int = 32,
    shuffle: bool = True,
    build_marginals: str | None = None,
    evolve_with_unitary=None,
):
    """Eagerly materialise the full training dataset as a list of batched tensor networks.

    Returns ``(batched_hyper_tns, batched_params_tns, batched_skeletons_tns)``,
    where ``params_tns`` and ``skeletons_tns`` are the JAX-compatible packed
    versions used by `optimize_tn`.
    """
    counts = experiments_to_tensors(experiments)

    if isinstance(build_marginals, str):
        print(f"> Generating marginals assuming {build_marginals} causal structure")  # noqa: T201
        counts = lightcones_marginal_experiments(counts, lightcone_structure=build_marginals)

    print(f"Shots = {counts.shape}")  # noqa: T201
    batched_experiments = batch_experiments(counts, batch_size=batch_size, shuffle=shuffle)

    batched_hyper_tns = []
    batched_params_tns = []
    batched_skeletons_tns = []

    print("Creating tensor network representation of batches and evolving them")  # noqa: T201
    for batch in tqdm(batched_experiments):
        batch_tn_states, batch_tn_povms = experiments_to_hyper_tn(batch, states, povms)
        if evolve_with_unitary is not None:
            batch_tn_states = evolve_hyper_tn(batch_tn_states, evolve_with_unitary)

        states_params, states_skeleton = qtn.pack(batch_tn_states)
        povms_params, povms_skeleton = qtn.pack(batch_tn_povms)

        batched_hyper_tns.append((batch_tn_states, batch_tn_povms))
        batched_params_tns.append((states_params, povms_params))
        batched_skeletons_tns.append((states_skeleton, povms_skeleton))

    return batched_hyper_tns, batched_params_tns, batched_skeletons_tns
