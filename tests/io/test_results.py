"""Round-trip test for `save_results` / `load_results`."""

from __future__ import annotations

import numpy as np

from tn_noise_learning.io import load_results, save_results


def test_save_and_load_results_roundtrip(tmp_path):
    res = {
        "train_loss": [0.5, 0.3, 0.1],
        "test_loss": [0.6, 0.4, 0.2],
        "nll": [0.5, 0.3, 0.1],
        "distance_to_tp": [0.05, 0.02, 0.01],
        "lpdo_trace": [4.0, 4.0, 4.0],
        "frob_distance": [0.5, 0.2, 0.05],
        "probs": np.array([0.1, 0.9, 0.5], dtype=complex),
        "params": [{0: np.eye(2, dtype=complex), 1: np.eye(2, dtype=complex)}],
    }
    metadata = {"seed": 42, "noise_type": "identity"}
    path = save_results(res, file_name="smoke", file_folder=str(tmp_path), metadata=metadata)
    assert path.exists()

    loaded = load_results(path)
    assert loaded["noise_type"] == "identity"
    np.testing.assert_allclose(loaded["train_loss"], res["train_loss"])
    np.testing.assert_allclose(loaded["params"][0][0], res["params"][0][0])
