"""Top-level data generation routines for synthetic QPT experiments.

This module wraps `build_qpt_mpo` + `simulate_experiments` (or their SPAM-aware
counterparts) into one-shot functions and provides a command-line interface.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import qiskit as qk
import qiskit.qasm2 as qasm2

from ..noise_models.channels import select_noise_channel
from .sampling import flatten_list, local_preps_and_meas
from .simulator import build_qpt_mpo, simulate_experiments


def _prepare_strategy(
    strategy: str, num_qubits: int, num_states: int, num_shots: int, shots_per_setting: int
) -> tuple[int, str, list | None]:
    """Compute ``(num_shots, locality, preps_and_meas)`` for the chosen sampling strategy."""
    if strategy == "local":
        locality = 3
        num_settings = (num_states * 3) ** locality
        shots_per_setting = int(np.ceil(num_shots / num_settings))
        num_shots = num_settings * shots_per_setting
        preps_and_meas = local_preps_and_meas(num_qubits, locality=locality, num_shots=num_shots, num_states=num_states)
        return len(preps_and_meas), str(locality), preps_and_meas

    if strategy == "global":
        return num_shots, "", None

    if strategy == "global_repeated":
        preps_and_meas = [
            [(np.random.randint(0, num_states, size=num_qubits).tolist(),
              np.random.randint(0, 3, size=num_qubits).tolist())] * shots_per_setting
            for _ in range(num_shots // shots_per_setting)
        ]
        preps_and_meas = flatten_list(preps_and_meas)
        return len(preps_and_meas), "", preps_and_meas

    raise ValueError(f"Sampling strategy {strategy} not implemented.")


def generate_data(
    qc: qk.QuantumCircuit,
    noise_str: str = "identity",
    num_shots: int = 1000,
    shots_per_setting: int = 1,
    input_states: str = "sic_states",
    strategy: str = "global_repeated",
    joblib_workers: int = -1,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""Generate a synthetic QPT dataset by tensor-network simulation.

    Args:
        qc: ideal unitary circuit to characterise.
        noise_str: identifier passed to
            [`select_noise_channel`][tn_noise_learning.noise_models.channels.select_noise_channel].
        num_shots: total measurement shots.
        shots_per_setting: number of identical shots per preparation/measurement setting.
        input_states: ``"sic_states"`` (4 states/qubit) or ``"oc_pauli_eigenstates"`` (6).
        strategy: ``"local"``, ``"global"``, or ``"global_repeated"``.
        joblib_workers: parallelism (``-1`` ⇒ all cores).
        seed: numpy RNG seed for reproducibility.

    Returns:
        Dict ready for JSON serialisation, with keys: ``qc`` (QASM2),
        ``noise``, ``locality``, ``num_shots``, ``num_settings``,
        ``shots_per_setting``, ``strategy``, ``states``, ``povms``, ``counts``.
    """
    if seed is not None:
        np.random.seed(seed)

    num_qubits = qc.num_qubits
    noise_mpo = select_noise_channel(noise_str, num_qubits, tensor_type="mpo")

    if input_states == "oc_pauli_eigenstates":
        num_states = 6
    elif input_states == "sic_states":
        num_states = 4
    else:
        raise ValueError(f"Type of input states {input_states} not available.")

    num_shots, locality, preps_and_meas = _prepare_strategy(
        strategy, num_qubits, num_states, num_shots, shots_per_setting
    )

    qpt_mpo, states, povms = build_qpt_mpo(qc_unitary=qc, noise_mpo=noise_mpo, input_states=input_states)
    experiments = simulate_experiments(
        qpt_mpo, num_shots=num_shots, preps_and_meas=preps_and_meas, joblib_workers=joblib_workers
    )

    return {
        "qc": qasm2.dumps(qc),
        "noise": noise_str,
        "locality": locality,
        "num_shots": num_shots,
        "num_settings": num_shots // shots_per_setting,
        "shots_per_setting": shots_per_setting,
        "strategy": strategy,
        "states": states.astype("str").tolist(),
        "povms": povms.astype("str").tolist(),
        "counts": experiments,
    }


def save_dataset(res: dict[str, Any], file_folder: str | Path, file_name_stem: str) -> Path:
    """Save a generated QPT dataset to JSON. Appends today's date to the filename."""
    folder = Path(file_folder)
    folder.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    path = folder / f"{file_name_stem}_{today}.json"
    with path.open("w") as f:
        json.dump(res, f, indent=4)
    return path


def _default_cnot_chain_circuit(num_qubits: int) -> qk.QuantumCircuit:
    """Build a layer of nearest-neighbour CNOTs on adjacent qubit pairs ``(0, 1), (2, 3), …``."""
    qc = qk.QuantumCircuit(num_qubits)
    for q in range(0, num_qubits - 1, 2):
        qc.cx(q, q + 1)
    return qc


def _cli() -> None:
    """Command-line driver: ``python -m tn_noise_learning.qpt.data_generation``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--noise", default="identity", help="Noise preset name (see select_noise_channel).")
    parser.add_argument("--num-shots", type=int, default=1000)
    parser.add_argument("--shots-per-setting", type=int, default=1)
    parser.add_argument(
        "--input-states", choices=["sic_states", "oc_pauli_eigenstates"], default="sic_states"
    )
    parser.add_argument("--strategy", choices=["local", "global", "global_repeated"], default="global_repeated")
    parser.add_argument("--output-dir", default="data/simulated_experiments")
    parser.add_argument("--joblib-workers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    qc = _default_cnot_chain_circuit(args.num_qubits)
    res = generate_data(
        qc=qc,
        noise_str=args.noise,
        num_shots=args.num_shots,
        shots_per_setting=args.shots_per_setting,
        input_states=args.input_states,
        strategy=args.strategy,
        joblib_workers=args.joblib_workers,
        seed=args.seed,
    )
    stem = (
        f"cxeven_nq{args.num_qubits}_{args.noise}_ns{res['num_shots']}_"
        f"{args.strategy}{res['locality']}_{args.input_states}"
    )
    out_path = save_dataset(res, args.output_dir, stem)
    print(f"Saved {res['num_shots']} shots to {out_path}")  # noqa: T201


if __name__ == "__main__":
    _cli()
