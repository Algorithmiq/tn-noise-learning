from __future__ import annotations

import pytest

from tn_noise_learning.history import History

pytest.importorskip("matplotlib")


def test_plot_history_panels(tmp_path):
    history = History(num_sites=2, best_epoch=2)
    for epoch in range(3):
        history.append(
            {
                "train_loss": 1.0 / (epoch + 1),
                "test_loss": 1.0,
                "nll": 1.0,
                "tp_penalty": 0.1,
                "trace": 4.0,
                "custom": 2.0,
            }
        )
    fig = history.plot(tmp_path / "history.png")
    assert len(fig.axes) == 5
    assert (tmp_path / "history.png").exists()

    with_ideal = History(num_sites=2)
    with_ideal.append({"train_loss": 1.0, "nll": 1.0, "tp_penalty": 0.1, "trace": 4.0, "distance_to_ideal": 1e-3})
    assert len(with_ideal.plot().axes) == 5
