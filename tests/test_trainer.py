from __future__ import annotations

import numpy as np
import pytest

from tn_noise_learning.lpdo import LPDO
from tn_noise_learning.topology import Topology
from tn_noise_learning.trainer import Trainer, TrainingConfig, project_trace_preserving


def small_config(**overrides):
    defaults = {"epochs": 2, "batch_size": 100, "tp_warmstart_steps": 20, "log_every": 1}
    return TrainingConfig(**(defaults | overrides))


def test_project_trace_preserving_reduces_penalty():
    model = LPDO.random(Topology.square(2, 2), bond_dim=2, kraus_dim=2, seed=0)
    projected = project_trace_preserving(model, steps=50)
    assert abs(projected.trace() - 16) < abs(model.trace() - 16)


def test_fit_smoke(chain2_dataset):
    train, test = chain2_dataset.split(0.1, seed=0)
    model = LPDO.random(Topology.chain(2), bond_dim=2, kraus_dim=2, seed=0)
    before = [t.copy() for t in model.tensors]
    history = Trainer(small_config()).fit(model, train, test)
    assert history.num_epochs == 2
    assert set(history.metrics) >= {"train_loss", "test_loss", "nll", "tp_penalty", "trace", "distance_to_ideal"}
    assert history.best_epoch in (1, 2)
    assert history.params == []
    assert any(not np.allclose(a, b) for a, b in zip(before, model.tensors, strict=True))


def test_snapshots_custom_metric_and_loss_term(chain2_dataset):
    model = LPDO.random(Topology.chain(2), bond_dim=2, kraus_dim=2, seed=1)
    config = small_config(
        snapshot_params=True,
        metrics={"bond": lambda m: m.bond_dim},
        extra_loss_terms={"zero": (lambda tn, probs: 0.0 * probs.sum(), 0.5)},
        tp_weight=lambda step: 2.0 if step < 3 else 1.0,
    )
    history = Trainer(config).fit(model, chain2_dataset)
    assert len(history.params) == 2
    assert history.metrics["bond"] == [2.0, 2.0]
    assert history.metrics["zero"] == [0.0, 0.0]


def test_early_stopping(chain2_dataset):
    model = LPDO.random(Topology.chain(2), bond_dim=2, kraus_dim=2, seed=2)
    history = Trainer(small_config(epochs=10, patience=1, tolerance=1e9)).fit(model, chain2_dataset)
    assert history.num_epochs == 2


def test_unknown_monitor_raises(chain2_dataset):
    model = LPDO.random(Topology.chain(2), bond_dim=2, kraus_dim=2, seed=3)
    with pytest.raises(KeyError, match="not a recorded metric"):
        Trainer(small_config(monitor="nope")).fit(model, chain2_dataset)
