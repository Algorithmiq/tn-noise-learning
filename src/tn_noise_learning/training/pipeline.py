"""High-level `train` orchestration that reproduces sabrina's noise_learning.py script.

Provides one function that runs the full pipeline end-to-end on a saved
QPT-experiment JSON file: build ground-truth LPDO, initialise a random LPDO,
warm-start it onto the TP manifold, batch the data, run SGD, and return the
training trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import optax
import qiskit as qk
import quimb as qu
from jax import config

from ..io.results import save_results
from ..noise_models.channels import select_noise_channel
from ..tn_core.lpdo_utils import build_lpdo, lpdo_to_mpo, trace_lpdo
from .dataset import create_training_dataset
from .optimizer import optimize_tn, optimize_trace_preservation

config.update("jax_enable_x64", True)


def train(
    data_file: str | Path,
    *,
    spam: bool = False,
    seed: int = 34343400,
    virtual_bd: int = 4,
    kraus_bd: int = 4,
    batch_size: int = 250,
    epochs: int = 150,
    lr: str | float = "exponential_decay",
    build_marginals: str | None = None,
    tp_warmstart_steps: int = 2500,
    num_thread_workers: int = 3,
    save: bool = True,
    output_dir: str | Path | None = None,
    use_qdt_effects: bool = False,
) -> dict[str, Any]:
    """Run the full LPDO training pipeline on a saved QPT experiment file.

    This single function folds the original ``noise_learning.py`` and
    ``noise_learning_spam.py`` scripts. The only differences between the two
    were default I/O paths and seeds — pass ``spam=True`` to get the SPAM
    defaults, or set the paths/seed explicitly to override.

    Args:
        data_file: path (or stem) of the experiment JSON file.
        spam: if True, look in ``data/simulated_experiments/spam/`` and write
            to ``data/optimisation_data/spam/`` by default.
        seed: RNG seed (defaults: 34343400 non-SPAM, 16 SPAM).
        virtual_bd: virtual bond dimension χ of the trainable LPDO.
        kraus_bd: Kraus dimension K of the trainable LPDO.
        batch_size: batch size for the SGD loop.
        epochs: maximum number of training epochs.
        lr: ``"exponential_decay"`` (default) or a constant learning rate.
        build_marginals: ``"brickwall"`` to add lightcone-marginal augmentation,
            ``None`` to skip.
        tp_warmstart_steps: L-BFGS-B steps used to push the initial random LPDO
            onto the TP manifold.
        num_thread_workers: ``quimb.core._NUM_THREAD_WORKERS`` value.
        save: if True, write `optimize_tn`'s output to JSON in ``output_dir``.
        output_dir: override the default save folder.
        use_qdt_effects: SPAM only — if True, use ``exp_data["qdt_effects"]``
            instead of ``exp_data["povms"]`` for the measurement effects.

    Returns:
        Dict produced by `optimize_tn` (training metrics + final parameters).
    """
    if spam and seed == 34343400:
        seed = 16

    np.random.seed(seed)
    _ = jax.random.PRNGKey(seed)
    qu.core._NUM_THREAD_WORKERS = num_thread_workers

    data_file = str(data_file)
    if not data_file.endswith(".json"):
        data_file = data_file + ".json"
    data_path = Path(data_file)
    if not data_path.is_absolute() and not data_path.exists():
        prefix = "data/simulated_experiments/spam/" if spam else "data/simulated_experiments/"
        data_path = Path(prefix) / data_file

    with data_path.open() as f:
        exp_data = json.load(f)

    qasm_qc, noise_str = exp_data["qc"], exp_data["noise"]
    print(f"Noise model: {noise_str}")  # noqa: T201

    qc = qk.QuantumCircuit.from_qasm_str(qasm_qc)
    num_qubits = qc.num_qubits
    print(qc)  # noqa: T201

    true_lpdo = select_noise_channel(noise_str, tensor_type="lpdo", num_qubits=num_qubits)
    true_mpo = select_noise_channel(noise_str, tensor_type="mpo", num_qubits=num_qubits)
    print(f"||MPO[simulation] - MPO[LPDO]|| = {lpdo_to_mpo(true_lpdo).distance(true_mpo)}")  # noqa: T201

    init_lpdo = build_lpdo(
        num_qubits=num_qubits, virtual_bd=virtual_bd, kraus_bd=kraus_bd, init_method="normal_complex"
    )
    init_lpdo, _tnopt = optimize_trace_preservation(
        init_lpdo, optimizer="L-BFGS-B", steps=tp_warmstart_steps, return_lpdo=True, plot_results=False
    )
    print(f"Tr[LPDO] = {trace_lpdo(init_lpdo)} (target = {2**num_qubits})")  # noqa: T201

    counts = exp_data["counts"]
    states = exp_data["states"]
    povms = exp_data["qdt_effects"] if (spam and use_qdt_effects) else exp_data["povms"]

    batched_hyper_tns, batched_params_tns, batched_skeletons_tns = create_training_dataset(
        counts,
        states,
        povms,
        batch_size=batch_size,
        shuffle=True,
        build_marginals=build_marginals,
        evolve_with_unitary=qc,
    )

    exp_dataset = batched_params_tns, batched_skeletons_tns[0]

    num_batches = len(batched_hyper_tns)
    tot_shots = batch_size * num_batches
    num_test_batches = int(min(max(1, ((0.1 * tot_shots) / batch_size)), 50))
    num_train_batches = num_batches - num_test_batches

    if lr == "exponential_decay":
        schedule = optax.exponential_decay(
            init_value=1e-2,
            transition_steps=min(500, num_train_batches),
            decay_rate=0.9,
            transition_begin=num_train_batches,
        )
        optimizer = optax.adam(learning_rate=schedule)
    else:
        optimizer = optax.adam(learning_rate=lr)

    kraus_tn = init_lpdo.select("V")
    res = optimize_tn(exp_dataset, kraus_tn, optimizer, true_lpdo=true_lpdo, epochs=epochs)

    if save:
        if output_dir is None:
            output_dir = "data/optimisation_data/spam/" if spam else "data/optimisation_data/"
        save_results(
            res,
            file_name=data_path.stem,
            file_folder=str(output_dir),
            metadata={
                "batch_size": batch_size,
                "build_marginal": build_marginals,
                "lr": lr,
                "seed": seed,
                "noise_type": noise_str,
                "data_file": data_path.stem,
                "virtual_bd": virtual_bd,
                "kraus_bd": kraus_bd,
            },
        )

    return res
