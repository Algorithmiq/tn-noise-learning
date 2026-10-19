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

"""Shared fixtures for the tn_noise_learning test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tn_noise_learning.channels import depolarizing
from tn_noise_learning.simulate import sample_shots
from tn_noise_learning.topology import Topology

if TYPE_CHECKING:
    from tn_noise_learning.dataset import NoiseDataset


@pytest.fixture(scope="session")
def chain2_dataset() -> NoiseDataset:
    """One thousand shots of a depolarizing channel on two qubits, with the ideal channel attached."""
    return sample_shots(depolarizing(Topology.chain(2), 0.1), num_settings=40, shots_per_setting=25, seed=0)
