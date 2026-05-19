"""Serialise / deserialise optimised LPDO results to JSON.

Complex-valued numpy arrays don't survive standard ``json.dump``. The helpers
here convert to/from string representations so the trajectories and Kraus
parameters produced by [`optimize_tn`][tn_noise_learning.training.optimizer.optimize_tn]
can round-trip through disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _make_json_compatible(x: Any) -> Any:
    """Convert a numpy array (possibly complex) into a nested list of strings."""
    return np.array(x, dtype=str).tolist()


def save_results(
    res: dict,
    file_name: str,
    file_folder: str | Path = "data/optimisation_data/",
    metadata: dict | None = None,
) -> Path:
    """Save the dict returned by `optimize_tn` to ``<file_folder>/<file_name>_seed<seed>.json``.

    Args:
        res: training metrics dict (output of `optimize_tn`).
        file_name: stem of the output file (no extension, no seed suffix).
        file_folder: target directory (created if missing).
        metadata: optional metadata to merge into the saved dict (typically
            ``batch_size``, ``lr``, ``seed``, ``noise_type``, etc.).

    Returns:
        Path of the written file.
    """
    folder = Path(file_folder)
    folder.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, Any] = dict(metadata or {})

    for k in ("train_loss", "nll", "distance_to_tp", "lpdo_trace", "frob_distance", "test_loss"):
        if k in res:
            save_dict[k] = _make_json_compatible(res[k])

    if "probs" in res:
        save_dict["probs"] = _make_json_compatible(res["probs"])

    if "params" in res:
        save_dict["params"] = [
            {k: np.array(v, dtype=str).tolist() for k, v in p.items()} for p in res["params"]
        ]

    seed = save_dict.get("seed", "")
    suffix = f"_seed{seed}" if seed != "" else ""
    path = folder / f"{file_name}{suffix}.json"

    with path.open("w") as f:
        json.dump(save_dict, f, indent=4)
    return path


def load_results(path: str | Path) -> dict[str, Any]:
    """Load a results JSON file written by `save_results`.

    Complex-typed entries (``probs``, ``params``) are converted back to numpy
    arrays of dtype ``complex``.
    """
    with Path(path).open() as f:
        data = json.load(f)

    if "probs" in data:
        data["probs"] = np.array(data["probs"], dtype=complex)
    if "params" in data:
        # JSON serialises dict keys as strings; restore the original int keys.
        data["params"] = [
            {int(k): np.array(v, dtype=complex) for k, v in p.items()} for p in data["params"]
        ]

    for k in ("train_loss", "nll", "distance_to_tp", "lpdo_trace", "frob_distance", "test_loss"):
        if k in data:
            arr = np.array(data[k], dtype=complex)
            if np.all(arr.imag == 0):
                arr = arr.real
            data[k] = arr

    return data
