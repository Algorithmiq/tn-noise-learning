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
