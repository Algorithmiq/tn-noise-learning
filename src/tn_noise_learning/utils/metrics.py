"""Channel comparison metrics on top of the LPDO helpers in `tn_core.lpdo_utils`."""

from __future__ import annotations

import numpy as np

from ..tn_core.lpdo_model import LPDOSuperOperator1D
from ..tn_core.lpdo_utils import lpdo_fidelity, trace_lpdo

__all__ = ["channel_frob_distance", "lpdo_fidelity", "trace_lpdo"]


def channel_frob_distance(lpdo_a: LPDOSuperOperator1D, lpdo_b: LPDOSuperOperator1D, backend: str = "jax") -> float:
    r"""Frobenius distance between two LPDOs, ``‖Λ_a − Λ_b‖_F / 2^n``.

    Differs from [`lpdo_fidelity`][tn_noise_learning.tn_core.lpdo_utils.lpdo_fidelity]
    in that it returns the linear distance (not the squared one).
    """
    import quimb.tensor as qtn

    num_qubits = lpdo_a.L
    return float(np.real(qtn.tensor_network_distance(lpdo_a, lpdo_b, backend=backend)) / (2**num_qubits))
