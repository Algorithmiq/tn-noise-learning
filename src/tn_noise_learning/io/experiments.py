"""JSON I/O helpers for QPT experiment datasets.

The standard sabrina JSON layout stored under ``data/simulated_experiments/``
is::

    {
        "qc": "<QASM2 string>",
        "noise": "<noise preset name>",
        "locality": "...",
        "num_shots": int,
        "num_settings": int,
        "shots_per_setting": int,
        "strategy": "global" | "local" | "global_repeated",
        "states": [[[str, ...], ...], ...],   # complex matrices as strings
        "povms":  [[[str, ...], ...], ...],
        "counts": { "Exp. 1": {input_str: outcome_str}, ... }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_experiments(path: str | Path) -> dict[str, Any]:
    """Load a QPT experiment JSON file.

    Returns the raw dictionary as written by
    [`save_dataset`][tn_noise_learning.qpt.data_generation.save_dataset]; the
    ``states`` and ``povms`` fields are converted back from their string
    representation to complex numpy arrays.
    """
    with Path(path).open() as f:
        data = json.load(f)

    if "states" in data:
        data["states_array"] = np.array(data["states"], dtype=complex)
    if "povms" in data:
        data["povms_array"] = np.array(data["povms"], dtype=complex)
    return data


def save_experiments(data: dict[str, Any], path: str | Path) -> None:
    """Write an experiment dict to JSON, stringifying any complex arrays under ``states``/``povms``."""
    payload = dict(data)
    if "states" in payload and isinstance(payload["states"], np.ndarray):
        payload["states"] = payload["states"].astype(str).tolist()
    if "povms" in payload and isinstance(payload["povms"], np.ndarray):
        payload["povms"] = payload["povms"].astype(str).tolist()
    payload.pop("states_array", None)
    payload.pop("povms_array", None)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(payload, f, indent=4)
