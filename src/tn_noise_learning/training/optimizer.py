"""JAX/optax-based optimisation of LPDO parameters."""

from __future__ import annotations

from collections import deque
from functools import partial

import jax
import numpy as np
import optax
import quimb.tensor as qtn
from jax import numpy as jnp
from tqdm import tqdm

from .losses import compute_metrics, loss_fn


def optimize_trace_preservation(
    lpdo,
    steps: int = 100,
    optimizer: str = "L-BFGS-B",
    return_lpdo: bool = True,
    plot_results: bool = True,
):
    r"""Project an LPDO onto the trace-preserving manifold by minimising the TP penalty alone.

    Used to warm-start the main training loop with an LPDO that is already
    close to a physically valid channel.
    """
    from ..tn_core.lpdo_utils import build_lpdo

    def tp_penalty_local(kraus_tn):
        reindex_outer = {
            kraus_tn.outer_upper_ind_id.format(_): kraus_tn.outer_lower_ind_id.format(_) for _ in range(kraus_tn.L)
        }
        reindex_inner = {
            kraus_tn.inner_upper_ind_id.format(_): kraus_tn.inner_lower_ind_id.format(_) for _ in range(kraus_tn.L)
        }
        kraus_tn_dag = (kraus_tn.H).reindex(reindex_outer | reindex_inner)
        lpdo_local = qtn.TensorNetwork([kraus_tn, kraus_tn_dag])
        return jnp.real(
            qtn.tensor_network_distance(lpdo_local.reindex(reindex_outer), id_mpo, backend="jax")
            / (2 ** (num_qubits / 2))
        )

    lpdo_tn = lpdo.copy()
    num_qubits = lpdo_tn.L
    kraus_tn = lpdo_tn.select(["V"])

    id_mpo = qtn.MPO_identity(num_qubits)
    id_mpo.set_params(
        {0: jnp.array([jnp.eye(2)])}
        | {key: jnp.array([[jnp.eye(2)]]) for key in range(1, num_qubits - 1)}
        | {num_qubits - 1: jnp.array([jnp.eye(2)])}
    )

    tnopt = qtn.TNOptimizer(
        kraus_tn,
        loss_fn=tp_penalty_local,
        optimizer=optimizer,
        autodiff_backend="jax",
        jit_fn=True,
        progbar=True,
    )
    opt_kraus = tnopt.optimize(steps)

    if plot_results:
        import matplotlib.pyplot as plt

        plt.plot(tnopt.losses, label=r"$\|\mathrm{Tr}[\Lambda] - \mathbb{I}\|_F / 2^{n/2}$")
        plt.yscale("log")
        plt.xlabel("Opt. steps")
        plt.ylabel("Trace preservation penalty")
        plt.legend()

    if return_lpdo:
        return build_lpdo(tn_data=opt_kraus.arrays), tnopt
    return opt_kraus, tnopt


def optimize_tn(experiments, kraus_tn, optimizer, epochs: int = 10, true_lpdo=None) -> dict:
    """Train the LPDO Kraus tensors via stochastic gradient descent on the combined loss.

    Args:
        experiments: tuple ``(batched_params, skeletons[0])`` from
            [`create_training_dataset`][tn_noise_learning.training.dataset.create_training_dataset].
        kraus_tn: initial Kraus tensor network (the ``V``-tagged half of an LPDO).
        optimizer: an ``optax`` optimiser instance.
        epochs: maximum number of epochs (loop short-circuits if test loss plateaus).
        true_lpdo: optional ground-truth LPDO for channel-fidelity tracking.

    Returns:
        Dict containing training trajectories under keys ``train_loss``,
        ``test_loss``, ``nll``, ``distance_to_tp``, ``lpdo_trace``,
        ``frob_distance``, ``params``, ``probs``.
    """
    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @partial(jax.jit, static_argnums=[4])
    def train_step(params, batch_states, batch_povms, opt_state, skeletons):
        (loss, vals), grads = loss_grad_fn(params, batch_states, batch_povms, skeletons)
        # For complex-valued losses, the minimisation direction is the negative *conjugate* gradient.
        grads = jax.tree_util.tree_map(lambda g: jnp.conj(g), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, vals

    kraus_params, kraus_skeleton = qtn.pack(kraus_tn)

    batched_hyper_tns_params, (batch_state_skeleton, batch_povm_skeleton) = experiments
    num_batches = len(batched_hyper_tns_params)
    batch_size = batch_state_skeleton.ind_size("exp")
    tot_shots = batch_size * num_batches

    num_test_batches = int(min(max(1, ((0.1 * tot_shots) / batch_size)), 50))
    test_batched_hyper_tns_params, train_batched_hyper_tns_params = (
        batched_hyper_tns_params[:num_test_batches],
        batched_hyper_tns_params[num_test_batches:],
    )

    print("Dataset:")  # noqa: T201
    print(f" > Training batches = {len(train_batched_hyper_tns_params)}")  # noqa: T201
    print(f" > Test batches = {len(test_batched_hyper_tns_params)}")  # noqa: T201
    print(f" > Batch size = {batch_size}\n")  # noqa: T201

    skeletons = (kraus_skeleton, batch_state_skeleton, batch_povm_skeleton)

    batches_to_average = min(len(train_batched_hyper_tns_params), 50)
    batch_loss = jax.jit(loss_fn, static_argnums=[3])
    train_loss_avg0 = np.mean(
        [batch_loss(kraus_params, s, p, skeletons)[0] for s, p in train_batched_hyper_tns_params[:batches_to_average]]
    )
    _tmp, (probs0, nll_loss0, tp_penalty_loss0) = batch_loss(
        kraus_params, train_batched_hyper_tns_params[0][0], train_batched_hyper_tns_params[0][1], skeletons
    )

    metrics_vals = compute_metrics(
        kraus_params, skeletons, test_data=test_batched_hyper_tns_params, true_lpdo=true_lpdo
    )

    metrics = {
        "frob_distance": [],
        "lpdo_trace": [],
        "test_loss": [],
        "train_loss": [train_loss_avg0],
        "nll": [nll_loss0.tolist()],
        "distance_to_tp": [tp_penalty_loss0.tolist()],
        "params": [kraus_params],
        "probs": [probs0.tolist()],
    }
    for key in metrics_vals:
        val = metrics_vals[key]
        metrics[key].append(val.tolist() if hasattr(val, "tolist") else val)

    opt_state = optimizer.init(kraus_params)

    nll_avg: deque = deque(maxlen=batches_to_average)
    train_loss_avg: deque = deque(maxlen=batches_to_average)
    distance_to_tp_avg: deque = deque(maxlen=batches_to_average)

    try:
        epoch = 1
        b = 0
        for epoch in range(1, epochs + 1):
            progress_bar = tqdm(train_batched_hyper_tns_params)
            for b, batch in enumerate(progress_bar):
                batch_in_state, batch_out_povm = batch
                kraus_params, opt_state, loss_val, (probs, nll_loss, tp_penalty_loss) = train_step(
                    kraus_params, batch_in_state, batch_out_povm, opt_state, skeletons
                )

                nll_avg.append(nll_loss.tolist())
                train_loss_avg.append(loss_val.tolist())
                distance_to_tp_avg.append(tp_penalty_loss.tolist())

                if b == (len(train_batched_hyper_tns_params) - 1):
                    metrics["params"].append(kraus_params)
                    metrics["probs"].append(probs.tolist())
                    metrics["nll"].append(np.mean(nll_avg))
                    metrics["train_loss"].append(np.mean(train_loss_avg))
                    metrics["distance_to_tp"].append(np.mean(distance_to_tp_avg))

                    metrics_vals = compute_metrics(
                        kraus_params, skeletons, test_data=test_batched_hyper_tns_params, true_lpdo=true_lpdo
                    )
                    for key in metrics_vals:
                        metrics[key].append(metrics_vals[key].tolist())

                progress_bar.set_description(
                    f"Epoch {epoch:2d}/{epochs} — Batch Loss: {loss_val:.6f} || "
                    f"Test Loss: {metrics_vals['test_loss']:.7f} — "
                    f"Frob. distance: {metrics_vals['frob_distance']:.7f} — "
                    f"Tr[LPDO]: {metrics_vals['lpdo_trace']:.2f}"
                )

            # Early-stopping: if the relevant metric over the last 5 epochs is essentially flat, stop.
            if epoch > 5:
                last_loss = metrics["frob_distance"][-5:] if true_lpdo is not None else metrics["test_loss"][-5:]
                if not np.all(np.isfinite(last_loss)):
                    print(
                        f"\n > STOPPING TRAINING: NaN/Inf in last 5 test losses. Stopped at epoch {epoch}, batch {b}."
                    )
                    break
                mean_diffs = np.abs([last_loss[k] - last_loss[k + 1] for k in range(len(last_loss) - 1)]).mean()
                if mean_diffs < 1e-7:
                    print(
                        f"\n > STOPPING TRAINING: metric flat over 5 epochs ({mean_diffs}). "
                        f"Stopped at epoch {epoch}, batch {b}."
                    )
                    break

    except KeyboardInterrupt:
        print(f"Training interrupted by user at epoch {epoch}, batch {b}.")  # noqa: T201

    return metrics
