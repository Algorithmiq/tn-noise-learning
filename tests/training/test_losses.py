"""Smoke tests for the loss helpers (no full training loop)."""

from __future__ import annotations

import numpy as np
from jax import numpy as jnp

from tn_noise_learning.training import kl_divergence


def test_kl_divergence_uniform_probs():
    probs = jnp.array([0.5, 0.5])
    # NLL of a uniform 2-outcome distribution: -log(0.5).
    assert np.isclose(float(kl_divergence(probs)), -np.log(0.5))


def test_kl_divergence_certain_outcome():
    probs = jnp.array([1.0, 1.0, 1.0])
    assert np.isclose(float(kl_divergence(probs)), 0.0)
