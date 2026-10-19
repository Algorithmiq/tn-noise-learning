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

"""Tensor-network learning of quantum noise channels.

Pipeline: [`Topology`][tn_noise_learning.Topology] → [`LPDO`][tn_noise_learning.LPDO] →
[`NoiseDataset`][tn_noise_learning.NoiseDataset] → [`Trainer`][tn_noise_learning.Trainer] →
[`History`][tn_noise_learning.History].
"""

from __future__ import annotations

from . import _jax  # noqa: F401  (must run before any JAX array is created)
from ._version import __version__
from .channels import PAULI_EIGENSTATES, depolarizing, pauli_channel, pauli_eigenstates, two_qubit_depolarizing
from .circuits import apply_circuit, circuit_network, product_state_network
from .dataset import Batch, NoiseDataset
from .history import History
from .losses import LossTerm, negative_log_likelihood, probabilities, tp_penalty
from .lpdo import LPDO, full_network
from .plotting import plot_history
from .simulate import outcome_distribution, sample_shots
from .topology import Topology
from .trainer import Trainer, TrainingConfig, project_trace_preserving

__all__ = [
    "LPDO",
    "PAULI_EIGENSTATES",
    "Batch",
    "History",
    "LossTerm",
    "NoiseDataset",
    "Topology",
    "Trainer",
    "TrainingConfig",
    "__version__",
    "apply_circuit",
    "circuit_network",
    "depolarizing",
    "full_network",
    "negative_log_likelihood",
    "outcome_distribution",
    "pauli_channel",
    "pauli_eigenstates",
    "plot_history",
    "probabilities",
    "product_state_network",
    "project_trace_preserving",
    "sample_shots",
    "tp_penalty",
    "two_qubit_depolarizing",
]
