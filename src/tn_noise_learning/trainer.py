"""Stochastic gradient descent on the Kraus tensors of an LPDO with optax and JAX."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
import quimb.tensor as qtn

from .circuits import circuit_network
from .dataset import NoiseDataset
from .history import History
from .losses import LossTerm, negative_log_likelihood, probabilities, tp_penalty
from .lpdo import LPDO, full_network

logger = logging.getLogger(__name__)

Schedule = float | Callable[[int], float]
Metric = Callable[[LPDO], float]


def evaluate_schedule(schedule: Schedule, step: int) -> float:
    """Value of a constant or step-dependent schedule."""
    return float(schedule(step)) if callable(schedule) else float(schedule)


@dataclass
class TrainingConfig:
    """Hyper-parameters of [`Trainer`][tn_noise_learning.trainer.Trainer].

    Args:
        epochs: maximum number of passes over the training set.
        batch_size: shots per gradient step; every batch has exactly this size.
        learning_rate: constant, optax schedule, or callable of the step index; defaults to an exponential decay
            from ``1e-2`` by a factor ``0.8`` every 200 steps down to ``1e-4``.
        optimizer: factory turning the learning rate into an optax optimizer.
        tp_weight: weight of the trace-preservation penalty, constant or callable of the step index. The trained
            trace exceeds ``2**n`` by about ``1/(2 tp_weight)`` relative; weights far above 1000 make Adam stiff.
        tp_warmstart_steps: L-BFGS-B steps moving the initial model onto the trace-preserving manifold (0 disables).
        tp_project_every: re-run the L-BFGS-B projection every this many steps (``None`` disables).
        tp_project_steps: L-BFGS-B steps per projection.
        extra_loss_terms: ``name -> (term, weight)`` added to the loss; a term maps ``(lpdo_tn, probs)`` to a scalar.
        metrics: ``name -> metric(model)`` evaluated on the current model after every epoch.
        monitor: metric used for early stopping and model selection.
        patience: epochs without improvement of ``monitor`` before training stops.
        tolerance: minimum decrease of ``monitor`` that counts as an improvement.
        test_fraction: held-out fraction used when ``fit`` receives no test set.
        seed: seed for the train/test split and batch shuffling.
        snapshot_params: record the Kraus tensors after every epoch in the history.
        log_every: log one line every this many epochs.
        contract_optimize: cotengra path optimizer for the loss contraction.
    """

    epochs: int = 50
    batch_size: int = 256
    learning_rate: Schedule = field(default_factory=lambda: optax.exponential_decay(1e-2, 200, 0.8, end_value=1e-4))
    optimizer: Callable[[Schedule], optax.GradientTransformation] = optax.adam
    tp_weight: Schedule = 100.0
    tp_warmstart_steps: int = 500
    tp_project_every: int | None = None
    tp_project_steps: int = 50
    extra_loss_terms: dict[str, tuple[LossTerm, Schedule]] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    monitor: str = "test_loss"
    patience: int = 10
    tolerance: float = 1e-4
    test_fraction: float = 0.1
    seed: int = 0
    snapshot_params: bool = False
    log_every: int = 1
    contract_optimize: str = "auto-hq"


def project_trace_preserving(model: LPDO, steps: int) -> LPDO:
    """Minimise the trace-preservation penalty alone with L-BFGS-B, starting from ``model``."""
    topology = model.topology
    optimizer = qtn.TNOptimizer(
        model.kraus_network(),
        loss_fn=lambda tn: tp_penalty(full_network(tn, topology)),
        optimizer="L-BFGS-B",
        autodiff_backend="jax",
        jit_fn=True,
        progbar=False,
    )
    return LPDO.from_network(topology, optimizer.optimize(steps))


class Trainer:
    """Fit an [`LPDO`][tn_noise_learning.lpdo.LPDO] to a [`NoiseDataset`][tn_noise_learning.dataset.NoiseDataset]."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def fit(
        self,
        model: LPDO,
        train: NoiseDataset,
        test: NoiseDataset | None = None,
        ideal_channel: LPDO | None = None,
    ) -> History:
        """Train ``model`` in place and return the training history.

        Args:
            model: initial LPDO; on return it holds the parameters of the best epoch.
            train: training shots; its circuit is applied inside the loss.
            test: held-out shots; split off ``train`` when omitted.
            ideal_channel: ground truth to report ``distance_to_ideal``; defaults to ``train.ideal_channel``.
        """
        cfg = self.config
        if test is None:
            train, test = train.split(cfg.test_fraction, cfg.seed)
        ideal = train.ideal_channel if ideal_channel is None else ideal_channel
        topology = model.topology
        circuit_tn = None if train.circuit is None else circuit_network(train.circuit)

        if cfg.tp_warmstart_steps > 0:
            model.tensors = project_trace_preserving(model, cfg.tp_warmstart_steps).tensors
        params, skeleton = qtn.pack(model.kraus_network())
        terms: dict[str, LossTerm] = {"nll": negative_log_likelihood, "tp_penalty": tp_penalty}
        terms |= {name: term for name, (term, _) in cfg.extra_loss_terms.items()}
        schedules: list[Schedule] = [1.0, cfg.tp_weight, *(w for _, w in cfg.extra_loss_terms.values())]
        optimizer = cfg.optimizer(cfg.learning_rate)

        def loss(params: dict, states: jnp.ndarray, povms: jnp.ndarray, weights: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            lpdo_tn = full_network(qtn.unpack(params, skeleton), topology)
            probs = probabilities(lpdo_tn, states, povms, circuit_tn, optimize=cfg.contract_optimize)
            values = jnp.stack([term(lpdo_tn, probs) for term in terms.values()])
            return jnp.sum(weights * values), values

        @jax.jit
        def step(params: dict, opt_state: optax.OptState, states: jnp.ndarray, povms: jnp.ndarray, weights: jnp.ndarray) -> tuple:
            (total, values), grads = jax.value_and_grad(loss, has_aux=True)(params, states, povms, weights)
            grads = jax.tree_util.tree_map(jnp.conj, grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, total, values

        evaluate_loss = jax.jit(loss)

        def current_model(params: dict) -> LPDO:
            return LPDO.from_network(topology, qtn.unpack(params, skeleton))

        def weights_at(step_index: int) -> jnp.ndarray:
            return jnp.array([evaluate_schedule(s, step_index) for s in schedules])

        history = History(num_sites=topology.num_sites)
        opt_state = optimizer.init(params)
        best_params, best_value, best_epoch, stale_epochs = params, np.inf, 0, 0
        global_step = 0
        try:
            for epoch in range(1, cfg.epochs + 1):
                sums = np.zeros(len(terms) + 1)
                num_batches = 0
                for batch in train.batches(cfg.batch_size, shuffle=True, seed=cfg.seed + epoch):
                    params, opt_state, total, values = step(params, opt_state, batch.states, batch.povms, weights_at(global_step))
                    global_step += 1
                    sums += np.concatenate([[float(total)], np.asarray(values)])
                    num_batches += 1
                    if cfg.tp_project_every and global_step % cfg.tp_project_every == 0:
                        projected = project_trace_preserving(current_model(params), cfg.tp_project_steps)
                        params = dict(zip(params, qtn.pack(projected.kraus_network())[0].values(), strict=True))

                weights = weights_at(global_step)
                test_batches = test.batches(min(cfg.batch_size, len(test)), shuffle=False, drop_last=False)
                test_loss = np.mean([float(evaluate_loss(params, b.states, b.povms, weights)[0]) for b in test_batches])
                model_now = current_model(params)
                metrics = {"train_loss": sums[0] / num_batches, "test_loss": test_loss}
                metrics |= {name: sums[i + 1] / num_batches for i, name in enumerate(terms)}
                metrics["trace"] = model_now.trace()
                if ideal is not None:
                    metrics["distance_to_ideal"] = model_now.distance(ideal)
                metrics |= {name: float(metric(model_now)) for name, metric in cfg.metrics.items()}
                history.append(metrics, model_now.tensors if cfg.snapshot_params else None)
                if epoch % cfg.log_every == 0:
                    logger.info("epoch %3d/%d  %s", epoch, cfg.epochs, "  ".join(f"{k}={v:.4g}" for k, v in metrics.items()))

                if cfg.monitor not in metrics:
                    raise KeyError(f"monitor {cfg.monitor!r} is not a recorded metric: {sorted(metrics)}")
                value = metrics[cfg.monitor]
                if not np.isfinite(value):
                    logger.warning("Stopping: %s is not finite at epoch %d.", cfg.monitor, epoch)
                    break
                if value < best_value:
                    stale_epochs = 0 if value < best_value - cfg.tolerance else stale_epochs + 1
                    best_params, best_value, best_epoch = params, value, epoch
                else:
                    stale_epochs += 1
                if stale_epochs >= cfg.patience:
                    logger.info("Stopping: no improvement of %s for %d epochs.", cfg.monitor, cfg.patience)
                    break
        except KeyboardInterrupt:
            logger.warning("Training interrupted at epoch %d; keeping the best model so far.", epoch)

        model.tensors = current_model(best_params).tensors
        history.best_epoch = best_epoch
        return history
