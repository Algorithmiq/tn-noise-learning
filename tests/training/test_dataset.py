"""Smoke tests for the experiment-to-tensor batching pipeline."""

from __future__ import annotations

import numpy as np

from tn_noise_learning.training import (
    batch_experiments,
    brickwall_marginals_inds,
    experiments_to_tensors,
)


def test_experiments_to_tensors_shape():
    fake = {
        "Exp. 1": {"012": "345"},
        "Exp. 2": {"123": "455"},
    }
    arr = experiments_to_tensors(fake)
    assert arr.shape == (2, 2, 3)
    np.testing.assert_array_equal(arr[0, 0], [0, 1, 2])
    np.testing.assert_array_equal(arr[1, 1], [4, 5, 5])


def test_batch_experiments_returns_uniform_batches():
    arr = np.zeros((10, 2, 4), dtype=int)
    batched = batch_experiments(arr, batch_size=4, shuffle=False)
    assert batched.shape[1] == 4


def test_brickwall_marginals_inds_smoke():
    inds = brickwall_marginals_inds(num_qubits=4)
    # For 4 qubits and a 2-qubit effect support we expect 3 measurement windows.
    assert len(inds) == 3
