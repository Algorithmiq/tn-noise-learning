"""Build the full QPT MPO and simulate measurement shots from it."""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from tqdm import tqdm

from ..tn_core.mpo_utils import circuit_to_mpo_ptm, identity_mpo_ptm
from .sampling import state_selector
from .states import ic_condition_check, labels_to_tensors, pauli_eigenstates, rho_to_ptm, sic_tetrahedron_states


def build_qpt_mpo(
    qc_unitary: QuantumCircuit | None,
    noise_mpo: qtn.MatrixProductOperator | None = None,
    tags=("NOISE",),
    input_states="oc_pauli_eigenstates",
    povms="oc_pauli_eigenstates",
):
    r"""Build the QPT tensor network from input states, a unitary circuit, a noise channel and POVMs.

    Layout::

           STATE    |   UNITARY   |    NOISE    |    POVMs
        ------------|-------------|-------------|------------
                    |             |             |
        --a--O--i-- | --i--O--j-- | --j--O--k-- | --k--O--b--

    Args:
        qc_unitary: Qiskit circuit for the ideal evolution (``None`` ⇒ identity).
        noise_mpo: MPO for the noise channel in PTM form (``None`` ⇒ identity).
        tags: tags attached to the noise tensors.
        input_states: ``"oc_pauli_eigenstates"`` (default), ``"sic_states"``, or
            an explicit ``(num_states, 2, 2)`` / ``(num_qubits, num_states, 2, 2)`` array.
        povms: same option set as ``input_states``, but for the measurement effects.

    Returns:
        ``(qpt_tn, single_qubit_input_states, single_qubit_povms)``.
    """
    num_qubits = qc_unitary.num_qubits

    if isinstance(input_states, str):
        if input_states == "oc_pauli_eigenstates":
            single_qubit_input_states = pauli_eigenstates()
        elif input_states == "sic_states":
            single_qubit_input_states = sic_tetrahedron_states(type="pauli_symmetric")
        else:
            raise ValueError(
                f"Input states {input_states} not defined; only 'oc_pauli_eigenstates' and 'sic_states'."
            )
    elif isinstance(input_states, list | np.ndarray):
        single_qubit_input_states = input_states
    else:
        raise ValueError(f"Invalid type for input_states: {type(input_states)}.")

    if not ic_condition_check(single_qubit_input_states):
        raise ValueError(f"Input states are not informationally complete:\n {single_qubit_input_states}.")

    if len(single_qubit_input_states.shape) == 3:
        single_qubit_input_states = np.array([single_qubit_input_states] * num_qubits)
    elif len(single_qubit_input_states.shape) != 4:
        raise ValueError(f"Weird shape for input_states: {single_qubit_input_states.shape}.")

    ptm_input_states = np.array([[rho_to_ptm(s) for s in qubit_states] for qubit_states in single_qubit_input_states])

    if isinstance(povms, str):
        if povms == "oc_pauli_eigenstates":
            single_qubit_povms = pauli_eigenstates()
        else:
            raise ValueError(f"POVMs {povms} not defined; only 'oc_pauli_eigenstates'.")
    elif isinstance(povms, list | np.ndarray):
        single_qubit_povms = povms
    else:
        raise ValueError(f"Invalid type for povms: {type(povms)}.")

    if not ic_condition_check(single_qubit_povms):
        raise ValueError(f"POVMs are not informationally complete:\n {single_qubit_povms}.")

    if len(single_qubit_povms.shape) == 3:
        single_qubit_povms = np.array([single_qubit_povms] * num_qubits)
    elif len(single_qubit_povms.shape) != 4:
        raise ValueError(f"Weird shape for povms: {single_qubit_povms.shape}.")

    ptm_povms = np.array([[rho_to_ptm(p) for p in qubit_povms] for qubit_povms in single_qubit_povms])

    states_mpo = qtn.MPO_product_operator(ptm_input_states, tags=["STATES"], upper_ind_id="a_{}", lower_ind_id="i_{}")
    povm_mpo = qtn.MPO_product_operator(ptm_povms, tags=["POVMs"], upper_ind_id="b_{}", lower_ind_id="k_{}")

    if qc_unitary is None:
        qc_mpo = identity_mpo_ptm(num_qubits)
    else:
        qc_mpo = circuit_to_mpo_ptm(qc_unitary, tags=["UNITARY"])

    if noise_mpo is None:
        noise_mpo = qtn.MPO_identity(num_qubits, phys_dim=4, upper_ind_id="k_{}", lower_ind_id="j_{}", tags=list(tags))
    else:
        noise_mpo.add_tag(list(tags))
        noise_mpo.reindex_upper_sites("k_{}", inplace=True)
        noise_mpo.reindex_lower_sites("j_{}", inplace=True)

    qpt_mpo = qtn.TensorNetwork([states_mpo, qc_mpo, noise_mpo, povm_mpo])
    return qpt_mpo, single_qubit_input_states, single_qubit_povms


def _sample_single_shot(
    preps_and_meas,
    qpt_mpo,
    num_qubits: int,
    num_states: int,
    size_povms: int,
    povm_normalisation: float,
    draw_tn: bool,
):
    """Inner loop body for `simulate_experiments` — sample one shot qubit-by-qubit."""
    input_state, meas_basis = preps_and_meas
    in_state = state_selector(input_state, state_space=num_states)

    measured_effect = None

    for idx_qub in range(num_qubits):
        qub_to_trace = list(range(idx_qub)) + list(range(idx_qub + 1, num_qubits))
        idx_to_trace = ["b_" + str(idx) for idx in qub_to_trace]
        tag_to_trace = ["I" + str(idx) for idx in qub_to_trace]
        all_keys = ["b_" + str(idx) for idx in range(num_qubits - 1)]
        all_tags = ["I" + str(idx) for idx in range(num_qubits - 1)]
        adjust_idx = dict(zip(all_keys, idx_to_trace, strict=True))
        adjust_tags = dict(zip(all_tags, tag_to_trace, strict=True))

        if measured_effect is None:
            tracing_tensor_data = np.ones(shape=(num_qubits - 1, size_povms)) / povm_normalisation
            measured_effect = []
        else:
            measured_effect_tn = labels_to_tensors(np.array(measured_effect))
            tracing_arrays = np.ones(shape=(num_qubits - 1 - idx_qub, size_povms)) / povm_normalisation
            tracing_tensor_data = np.vstack([measured_effect_tn, tracing_arrays])

        tr_MPS = qtn.MPS_product_state(tracing_tensor_data, site_ind_id="b_{}", tags="TRACING")
        tr_MPS.reindex(adjust_idx, inplace=True)
        tr_MPS.retag(adjust_tags, inplace=True)

        tn_transition_prob = qtn.TensorNetwork([in_state, qpt_mpo, tr_MPS])
        if draw_tn:
            tn_transition_prob.draw(
                color=["STATES", "POVMs", "NOISE", "UNITARY", "IN", "TRACING"], show_tags=False
            )

        probs = (tn_transition_prob.contract()).data
        try:
            probs = probs.reshape((size_povms,))
        except Exception as e:
            raise ValueError(f"Probabilities are not an array of size {size_povms}, got {probs.shape}.") from e

        if meas_basis is None:
            probs = np.abs(probs / probs.sum())
            sampled_effect = np.random.choice(range(size_povms), p=probs)
        else:
            probs = povm_normalisation * np.abs(probs / probs.sum())
            basis = meas_basis[idx_qub]
            sampled_effect = np.random.choice([2 * basis, 2 * basis + 1], p=probs[2 * basis : 2 * basis + 2])

        measured_effect.append(sampled_effect)

    in_state_str = "".join(map(str, list(input_state)))
    out_state_str = "".join(map(str, measured_effect))
    return in_state_str, out_state_str


def simulate_experiments(
    qpt_mpo,
    num_shots: int = 10,
    preps_and_meas=None,
    verbose: bool = False,
    draw_tn: bool = False,
    joblib_workers: int = 1,
) -> dict:
    r"""Sample measurement shots from a QPT MPO, qubit-by-qubit.

    The chain ``input → unitary → noise → POVM`` is contracted with a one-hot
    input state and a tracing MPS that fixes already-sampled qubits, yielding
    the marginal probability of the next qubit's outcome.

    Args:
        qpt_mpo: tensor network produced by `build_qpt_mpo`.
        num_shots: number of shots to sample (ignored if ``preps_and_meas`` is given).
        preps_and_meas: explicit list of ``(input_state, meas_basis)`` pairs.
        verbose: print progress.
        draw_tn: draw the contraction tensor network for each qubit (debug).
        joblib_workers: number of joblib workers (``1`` = serial, ``-1`` = all cores).

    Returns:
        Dict mapping ``"Exp. m"`` to ``{input_state_string: output_state_string}``.
    """
    num_qubits = int(len(qpt_mpo.shape) / 2)
    num_states = qpt_mpo["STATES"][0].ind_size("a_0")
    size_povms = qpt_mpo["POVMs"][0].ind_size("b_0")
    povm_normalisation = round(np.sum(qpt_mpo["POVMs"][0].data, axis=1)[0, 0] / np.sqrt(2))

    if preps_and_meas is None:
        input_states = np.random.randint(0, high=num_states, size=(num_shots, num_qubits))
        preps_and_meas = [[s, None] for s in input_states]

    results = Parallel(n_jobs=joblib_workers)(
        delayed(_sample_single_shot)(
            p_and_m, qpt_mpo, num_qubits, num_states, size_povms, povm_normalisation, draw_tn
        )
        for p_and_m in tqdm(preps_and_meas)
    )

    return {f"Exp. {m + 1}": {in_state: out_effect} for m, (in_state, out_effect) in enumerate(results)}
