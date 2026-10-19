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
