"""Shared fixtures for the tn_noise_learning test suite."""

from __future__ import annotations

import numpy as np
import pytest

from tn_noise_learning.channels import depolarizing
from tn_noise_learning.dataset import NoiseDataset
from tn_noise_learning.simulate import sample_shots
from tn_noise_learning.topology import Topology


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy Generator for reproducible smoke tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def chain2_dataset() -> NoiseDataset:
    """One thousand shots of a depolarizing channel on two qubits, with the ideal channel attached."""
    return sample_shots(depolarizing(Topology.chain(2), 0.1), num_settings=40, shots_per_setting=25, seed=0)
