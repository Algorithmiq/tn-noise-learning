"""Tensor-network learning of quantum noise channels.

Public API surface. Each subpackage exposes its own complete API; the names
re-exported here are the most common entry points for the end-to-end pipeline:

* `select_noise_channel` — build a noise channel by short string identifier.
* `build_lpdo`, `LPDOSuperOperator1D` — construct LPDO super-operators.
* `qc_to_mpo` — Qiskit circuit to unitary MPO.
* `build_qpt_mpo`, `simulate_experiments`, `generate_data` — QPT simulation.
* `train`, `optimize_tn` — JAX-based optimisation of LPDO parameters.
* `save_results`, `load_results` — JSON I/O for trained LPDOs.
"""

from __future__ import annotations

from .io import load_results, save_results
from .noise_models import select_noise_channel
from .qpt import (
    build_qpt_mpo,
    generate_data,
    pauli_eigenstates,
    sic_tetrahedron_states,
    simulate_experiments,
)
from .tn_core import (
    LPDOSuperOperator1D,
    build_lpdo,
    compose_lpdos,
    lpdo_fidelity,
    lpdo_to_mpo,
    qc_to_mpo,
)
from .training import kl_divergence, optimize_tn, train

__version__ = "0.1.0"

__all__ = [
    "LPDOSuperOperator1D",
    "__version__",
    "build_lpdo",
    "build_qpt_mpo",
    "compose_lpdos",
    "generate_data",
    "kl_divergence",
    "load_results",
    "lpdo_fidelity",
    "lpdo_to_mpo",
    "optimize_tn",
    "pauli_eigenstates",
    "qc_to_mpo",
    "save_results",
    "select_noise_channel",
    "sic_tetrahedron_states",
    "simulate_experiments",
    "train",
]
