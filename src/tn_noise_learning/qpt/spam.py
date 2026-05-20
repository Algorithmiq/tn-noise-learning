"""SPAM-aware QPT tensor network (separate prep and measurement error MPOs).

Variant of [`build_qpt_mpo`][tn_noise_learning.qpt.simulator.build_qpt_mpo]
that inserts independent state-preparation and measurement-error MPOs around
the ideal unitary + noise channel.
"""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from tqdm import tqdm

from ..tn_core.mpo_utils import circuit_to_mpo_ptm
from .states import ic_condition_check, labels_to_tensors, pauli_eigenstates, rho_to_ptm, sic_tetrahedron_states


def _combine_tns(
    mpo_after: qtn.MatrixProductOperator, mpo_before: qtn.MatrixProductOperator
) -> qtn.MatrixProductOperator:
    """Adapter for `quimb.tensor.apply_mpo` that combines two MPOs without contracting them.

    Acts like matrix multiplication: ``MPO = MPO_after @ MPO_before``.
    """
    A, B = mpo_after.copy(), mpo_before.copy()
    A.upper_ind_id = B.upper_ind_id
    tmp_ind = qtn.rand_uuid()
    B.upper_ind_id = tmp_ind + "{}"
    A.lower_ind_id = tmp_ind + "{}"
    AB = A | B
    return AB.view_as_(
        qtn.MatrixProductOperator, upper_ind_id=A.upper_ind_id, lower_ind_id=B.lower_ind_id, cyclic=False
    )


def build_qpt_tn(
    input_states="oc_pauli_eigenstates",
    povm="oc_pauli_eigenstates",
    qc_unitary: QuantumCircuit | None = None,
    prep_error_mpo: qtn.MatrixProductOperator | None = None,
    noise_mpo: qtn.MatrixProductOperator | None = None,
    meas_error_mpo: qtn.MatrixProductOperator | None = None,
):
    """Build a SPAM-aware QPT tensor network.

    The tensor product is ``STATES → prep_error → unitary → noise → meas_error → POVMs``.

    Args:
        input_states: ``"oc_pauli_eigenstates"`` or ``"sic_states"`` or explicit array.
        povm: same options as ``input_states``.
        qc_unitary: ideal evolution (``None`` ⇒ identity).
        prep_error_mpo: state-preparation error channel (``None`` ⇒ identity).
        noise_mpo: process noise channel.
        meas_error_mpo: measurement error channel (``None`` ⇒ identity).

    Returns:
        ``(qpt_mpo, qpt_setup, qpt_full_tn)``.
    """
    num_qubits = qc_unitary.num_qubits

    if isinstance(input_states, str):
        if input_states == "oc_pauli_eigenstates":
            single_qubit_input_states = pauli_eigenstates()
        elif input_states == "sic_states":
            single_qubit_input_states = sic_tetrahedron_states(type="pauli_symmetric")
        else:
            raise ValueError(f"Input states {input_states} not defined.")
    elif isinstance(input_states, list | np.ndarray):
        single_qubit_input_states = input_states
    else:
        raise ValueError(f"Invalid input_states type: {type(input_states)}.")

    if not ic_condition_check(single_qubit_input_states):
        raise ValueError("Input states are not informationally complete.")

    if len(single_qubit_input_states.shape) == 3:
        single_qubit_input_states = np.array([single_qubit_input_states] * num_qubits)
    elif len(single_qubit_input_states.shape) != 4:
        raise ValueError(f"Weird shape for input_states: {single_qubit_input_states.shape}.")

    ptm_input_states = np.array([[rho_to_ptm(s) for s in qubit_states] for qubit_states in single_qubit_input_states])
    ptm_input_states = ptm_input_states.swapaxes(2, 1)
    states_mpo = qtn.MPO_product_operator(ptm_input_states, tags=["STATES"])

    if isinstance(povm, str):
        if povm == "oc_pauli_eigenstates":
            single_qubit_povm = pauli_eigenstates()
        else:
            raise ValueError(f"POVM {povm} not defined.")
    elif isinstance(povm, list | np.ndarray):
        single_qubit_povm = povm
    else:
        raise ValueError(f"Invalid povm type: {type(povm)}.")

    if not ic_condition_check(single_qubit_povm):
        raise ValueError("POVM is not informationally complete.")

    if len(single_qubit_povm.shape) == 3:
        single_qubit_povm = np.array([single_qubit_povm] * num_qubits)
    elif len(single_qubit_povm.shape) != 4:
        raise ValueError(f"Weird shape for povm: {single_qubit_povm.shape}.")

    ptm_povm = np.array([[rho_to_ptm(p) for p in qubit_povm] for qubit_povm in single_qubit_povm])
    povm_mpo = qtn.MPO_product_operator(ptm_povm, tags=["POVM"])

    if prep_error_mpo is None:
        prep_error_mpo = qtn.MPO_identity(num_qubits, phys_dim=4, tags="PREP_ERROR")
    elif isinstance(prep_error_mpo, qtn.MatrixProductOperator):
        prep_error_mpo.add_tag("PREP_ERROR")
    else:
        raise TypeError(f"prep_error_mpo must be an MPO or None, got {type(prep_error_mpo)}.")

    if qc_unitary is None:
        qc_mpo = qtn.MPO_identity(num_qubits, phys_dim=4, tags=["UNITARY"])
    elif isinstance(qc_unitary, QuantumCircuit):
        qc_mpo = circuit_to_mpo_ptm(qc_unitary, tags=["UNITARY"])
    else:
        raise TypeError(f"qc_unitary must be a QuantumCircuit or None, got {type(qc_unitary)}.")

    if noise_mpo is None:
        noise_mpo = qtn.MPO_identity(num_qubits, phys_dim=4, tags="NOISE")
    elif isinstance(noise_mpo, qtn.MatrixProductOperator):
        noise_mpo.add_tag("NOISE")
    else:
        raise TypeError(f"noise_mpo must be an MPO or None, got {type(noise_mpo)}.")

    if meas_error_mpo is None:
        meas_error_mpo = qtn.MPO_identity(num_qubits, phys_dim=4, tags="MEAS_ERROR")
    elif isinstance(meas_error_mpo, qtn.MatrixProductOperator):
        meas_error_mpo.add_tag("MEAS_ERROR")
    else:
        raise TypeError(f"meas_error_mpo must be an MPO or None, got {type(meas_error_mpo)}.")

    qpt_mpo = states_mpo.copy()
    qpt_full_tn = states_mpo.copy()
    for mpo in [prep_error_mpo, qc_mpo, noise_mpo, meas_error_mpo, povm_mpo]:
        qpt_mpo = mpo._apply_mpo(qpt_mpo, compress=False, inplace=False)
        qpt_full_tn = _combine_tns(mpo, qpt_full_tn)

    qpt_setup = {
        "state_ind_id": states_mpo.lower_ind_id,
        "effect_ind_id": povm_mpo.upper_ind_id,
        "states_mpo": states_mpo,
        "povm_mpo": povm_mpo,
        "qc_mpo": qc_mpo,
        "prep_error_mpo": prep_error_mpo,
        "noise_mpo": noise_mpo,
        "meas_error_mpo": meas_error_mpo,
        "single_qubit_input_states": single_qubit_input_states,
        "single_qubit_povm": single_qubit_povm,
    }
    return qpt_mpo, qpt_setup, qpt_full_tn.view_as_(qtn.TensorNetwork1D)


def _sample_single_shot_spam(
    preps_and_meas,
    qpt_mpo,
    num_qubits: int,
    num_states: int,
    num_effects: int,
    povm_normalisation: float,
    state_ind_id: str,
    effect_ind_id: str,
    site_tag: str,
    draw_tn: bool,
):
    """Single-shot SPAM sampling — analogue of `_sample_single_shot` for the SPAM TN."""
    input_state, meas_basis = preps_and_meas
    alpha_one_hot = labels_to_tensors(np.array(input_state), state_space=num_states)
    in_state = qtn.MPS_product_state(alpha_one_hot, tags=None, site_ind_id=state_ind_id)

    measured_effect = None
    for idx_qub in range(num_qubits):
        qub_to_trace = list(range(idx_qub)) + list(range(idx_qub + 1, num_qubits))
        idx_to_trace = [effect_ind_id.format(idx) for idx in qub_to_trace]
        tag_to_trace = [site_tag.format(idx) for idx in qub_to_trace]
        all_keys = [effect_ind_id.format(idx) for idx in range(num_qubits - 1)]
        all_tags = [site_tag.format(idx) for idx in range(num_qubits - 1)]
        adjust_idx = dict(zip(all_keys, idx_to_trace, strict=True))
        adjust_tags = dict(zip(all_tags, tag_to_trace, strict=True))

        if measured_effect is None:
            tracing_tensor_data = np.ones(shape=(num_qubits - 1, num_effects)) / povm_normalisation
            measured_effect = []
        else:
            measured_effect_tn = labels_to_tensors(np.array(measured_effect))
            tracing_arrays = np.ones(shape=(num_qubits - 1 - idx_qub, num_effects)) / povm_normalisation
            tracing_tensor_data = np.vstack([measured_effect_tn, tracing_arrays])

        tr_MPS = qtn.MPS_product_state(tracing_tensor_data, site_ind_id=effect_ind_id, tags="TRACING")
        tr_MPS.reindex(adjust_idx, inplace=True)
        tr_MPS.retag(adjust_tags, inplace=True)

        tn_transition_prob = qtn.TensorNetwork([in_state, qpt_mpo, tr_MPS])
        if draw_tn:
            tn_transition_prob.draw(color=["STATES", "POVM", "NOISE", "UNITARY", "IN", "TRACING"], show_tags=False)

        probs = (tn_transition_prob.contract()).data
        try:
            probs = probs.reshape((num_effects,))
        except Exception as e:
            raise ValueError(f"Probabilities are not an array of size {num_effects}, got {probs.shape}.") from e

        if meas_basis is None:
            probs = np.abs(probs / probs.sum())
            sampled_effect = np.random.choice(range(num_effects), p=probs)
        else:
            probs = povm_normalisation * np.abs(probs / probs.sum())
            basis = meas_basis[idx_qub]
            sampled_effect = np.random.choice([2 * basis, 2 * basis + 1], p=probs[2 * basis : 2 * basis + 2])

        measured_effect.append(sampled_effect)

    in_state_str = "".join(map(str, list(input_state)))
    out_state_str = "".join(map(str, measured_effect))
    return in_state_str, out_state_str


def simulate_experiments_spam(
    qpt_mpo,
    qpt_setup,
    preps_and_meas=None,
    num_shots: int = 1,
    verbose: bool = False,
    draw_tn: bool = False,
    joblib_workers: int = 1,
) -> dict:
    """Simulate measurement shots from a SPAM-aware QPT TN built by [`build_qpt_tn`][tn_noise_learning.qpt.spam.build_qpt_tn]."""
    num_qubits = int(qpt_mpo.L)

    site_tag = qpt_mpo.site_tag_id
    state_ind_id = qpt_setup["state_ind_id"]
    effect_ind_id = qpt_setup["effect_ind_id"]

    num_states = qpt_mpo.ind_size(state_ind_id.format(0))
    num_effects = qpt_mpo.ind_size(effect_ind_id.format(0))

    povm_normalisation = [
        np.round(local_povm.data.squeeze().sum() / np.sqrt(2)) for local_povm in qpt_setup["povm_mpo"]
    ]
    if np.isclose(povm_normalisation[0] - povm_normalisation, 0).all():
        povm_normalisation = povm_normalisation[0]
    else:
        raise ValueError(
            f"POVM has different normalization per qubit; not yet implemented. povm_normalisation = {povm_normalisation}"
        )

    if verbose:
        print(  # noqa: T201
            f"Number of input states = {num_states}. Number of effects = {num_effects}. "
            f"Effects normalisation = {povm_normalisation}."
        )

    if preps_and_meas is None:
        input_states = np.random.randint(0, high=num_states, size=(num_shots, num_qubits))
        preps_and_meas = [[s, None] for s in input_states]

    results = Parallel(n_jobs=joblib_workers)(
        delayed(_sample_single_shot_spam)(
            p_and_m,
            qpt_mpo,
            num_qubits,
            num_states,
            num_effects,
            povm_normalisation,
            state_ind_id,
            effect_ind_id,
            site_tag,
            draw_tn,
        )
        for p_and_m in tqdm(preps_and_meas)
    )

    return {f"Exp. {m + 1}": {in_state: out_effect} for m, (in_state, out_effect) in enumerate(results)}
