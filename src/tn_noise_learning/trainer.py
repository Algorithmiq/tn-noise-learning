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

"""Stochastic gradient descent on the Kraus tensors of an LPDO with optax and JAX."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
import quimb.tensor as qtn

from .circuits import circuit_network
from .history import History
from .losses import LossTerm, negative_log_likelihood, probabilities, tp_penalty
from .lpdo import LPDO, full_network

if TYPE_CHECKING:
    from .dataset import NoiseDataset

logger = logging.getLogger(__name__)

Schedule = float | Callable[[int], float]
Metric = Callable[[LPDO], float]


def evaluate_schedule(schedule: Schedule, step: int) -> float:
    """Value of a constant or step-dependent schedule."""
    if isinstance(schedule, int | float):
        return float(schedule)
    return float(schedule(step))


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
    learning_rate: optax.ScalarOrSchedule = field(
        default_factory=lambda: optax.exponential_decay(1e-2, 200, 0.8, end_value=1e-4)
    )
    optimizer: Callable[[optax.ScalarOrSchedule], optax.GradientTransformation] = optax.adam
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


@dataclass
class _Selection:
    """Best-model tracking and early stopping on one monitored metric."""

    patience: int
    tolerance: float
    best_params: dict
    best_value: float = np.inf
    best_epoch: int = 0
    stale_epochs: int = 0

    def update(self, value: float, params: dict, epoch: int) -> bool:
        """Record an epoch; return ``True`` when training should stop."""
        if not np.isfinite(value):
            logger.warning("Stopping: monitored metric is not finite at epoch %d.", epoch)
            return True
        improved = value < self.best_value - self.tolerance
        if value < self.best_value:
            self.best_params, self.best_value, self.best_epoch = params, value, epoch
        self.stale_epochs = 0 if improved else self.stale_epochs + 1
        if self.stale_epochs >= self.patience:
            logger.info("Stopping: no improvement for %d epochs.", self.patience)
            return True
        return False


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
        if cfg.tp_warmstart_steps > 0:
            model.tensors = project_trace_preserving(model, cfg.tp_warmstart_steps).tensors

        params, skeleton = qtn.pack(model.kraus_network())
        rebuild = lambda p: LPDO.from_network(model.topology, qtn.unpack(p, skeleton))  # noqa: E731
        term_names, weights_at, step, evaluate_loss, opt_state = self._compile(model, params, skeleton, train)
        history = History(num_sites=model.num_sites)
        selection = _Selection(cfg.patience, cfg.tolerance, params)
        global_step = 0
        epoch = 0
        try:
            for epoch in range(1, cfg.epochs + 1):
                sums = np.zeros(len(term_names) + 1)
                for batch in train.batches(cfg.batch_size, shuffle=True, seed=cfg.seed + epoch):
                    params, opt_state, total, values = step(
                        params, opt_state, batch.states, batch.povms, weights_at(global_step)
                    )
                    sums += np.concatenate([[float(total)], np.asarray(values)])
                    global_step += 1
                    if cfg.tp_project_every and global_step % cfg.tp_project_every == 0:
                        projected = project_trace_preserving(rebuild(params), cfg.tp_project_steps)
                        params = dict(zip(params, qtn.pack(projected.kraus_network())[0].values(), strict=True))
                num_batches = len(train) // cfg.batch_size
                test_loss = np.mean(
                    [
                        float(evaluate_loss(params, b.states, b.povms, weights_at(global_step))[0])
                        for b in test.batches(min(cfg.batch_size, len(test)), shuffle=False, drop_last=False)
                    ]
                )
                metrics = {"train_loss": sums[0] / num_batches, "test_loss": float(test_loss)}
                metrics |= {name: sums[i + 1] / num_batches for i, name in enumerate(term_names)}
                current = rebuild(params)
                metrics |= self._model_metrics(current, ideal)
                history.append(metrics, current.tensors if cfg.snapshot_params else None)
                if epoch % cfg.log_every == 0:
                    logger.info(
                        "epoch %3d/%d  %s", epoch, cfg.epochs, "  ".join(f"{k}={v:.4g}" for k, v in metrics.items())
                    )
                if cfg.monitor not in metrics:
                    raise KeyError(f"monitor {cfg.monitor!r} is not a recorded metric: {sorted(metrics)}")
                if selection.update(metrics[cfg.monitor], params, epoch):
                    break
        except KeyboardInterrupt:
            logger.warning("Training interrupted at epoch %d; keeping the best model so far.", epoch)

        model.tensors = rebuild(selection.best_params).tensors
        history.best_epoch = selection.best_epoch
        return history

    def _model_metrics(self, model: LPDO, ideal: LPDO | None) -> dict[str, float]:
        """Trace, distance to the ideal channel when known, and user metrics."""
        metrics = {"trace": model.trace()}
        if ideal is not None:
            metrics["distance_to_ideal"] = model.distance(ideal)
        return metrics | {name: float(metric(model)) for name, metric in self.config.metrics.items()}

    def _compile(self, model: LPDO, params: dict, skeleton: qtn.TensorNetwork, train: NoiseDataset) -> tuple:
        """Build the jitted loss and update step.

        Returns the loss-term names, the weight schedule, the step function, the jitted loss and the optimizer state.
        """
        cfg = self.config
        topology = model.topology
        circuit_tn = None if train.circuit is None else circuit_network(train.circuit)
        terms: dict[str, LossTerm] = {"nll": negative_log_likelihood, "tp_penalty": tp_penalty}
        terms |= {name: term for name, (term, _) in cfg.extra_loss_terms.items()}
        schedules: list[Schedule] = [1.0, cfg.tp_weight, *(w for _, w in cfg.extra_loss_terms.values())]
        optimizer = cfg.optimizer(cfg.learning_rate)

        def loss(params: dict, states: jnp.ndarray, povms: jnp.ndarray, weights: jnp.ndarray) -> tuple:
            lpdo_tn = full_network(qtn.unpack(params, skeleton), topology)
            probs = probabilities(lpdo_tn, states, povms, circuit_tn, optimize=cfg.contract_optimize)
            values = jnp.stack([term(lpdo_tn, probs) for term in terms.values()])
            return jnp.sum(weights * values), values

        @jax.jit
        def step(
            params: dict, opt_state: optax.OptState, states: jnp.ndarray, povms: jnp.ndarray, weights: jnp.ndarray
        ) -> tuple:
            (total, values), grads = jax.value_and_grad(loss, has_aux=True)(params, states, povms, weights)
            grads = jax.tree_util.tree_map(jnp.conj, grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, total, values

        def weights_at(step_index: int) -> jnp.ndarray:
            return jnp.array([evaluate_schedule(s, step_index) for s in schedules])

        return list(terms), weights_at, step, jax.jit(loss), optimizer.init(params)
