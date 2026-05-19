"""Pytest fixtures + shared helpers for the tn_noise_learning test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy Generator for reproducible smoke tests."""
    return np.random.default_rng(seed=42)
