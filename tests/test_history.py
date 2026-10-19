from __future__ import annotations

import numpy as np

from tn_noise_learning.history import History


def test_round_trip(tmp_path):
    history = History(num_sites=2)
    history.append({"train_loss": 1.0, "trace": 3.9}, [np.ones((2, 2, 2)), np.zeros((2, 2, 2))])
    history.append({"train_loss": 0.5, "trace": 4.0}, [np.ones((2, 2, 2)), np.zeros((2, 2, 2))])
    history.best_epoch = 2
    history.save(tmp_path / "history.zarr")
    loaded = History.load(tmp_path / "history.zarr")
    assert loaded.metrics == {"train_loss": [1.0, 0.5], "trace": [3.9, 4.0]}
    assert loaded.best_epoch == 2
    assert loaded.num_sites == 2
    assert len(loaded.params) == 2
    np.testing.assert_array_equal(loaded.params[1][0], np.ones((2, 2, 2)))


def test_round_trip_without_params(tmp_path):
    history = History()
    history.append({"train_loss": 1.0})
    history.save(tmp_path / "history.zarr")
    loaded = History.load(tmp_path / "history.zarr")
    assert loaded.params == []
    assert loaded.best_epoch is None
