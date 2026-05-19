"""Dispatcher for named noise models.

[`select_noise_channel`][tn_noise_learning.noise_models.channels.select_noise_channel]
is the recommended public entry point — it returns the LPDO or MPO of a noise
channel given a short string identifier. The two tensor-type-specific modules
([`lpdo_channels`][tn_noise_learning.noise_models.lpdo_channels] and
[`mpo_channels`][tn_noise_learning.noise_models.mpo_channels]) provide the
underlying constructors directly.
"""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn

from ..tn_core.lpdo_model import LPDOSuperOperator1D
from ..tn_core.lpdo_utils import compose_lpdos
from . import lpdo_channels, mpo_channels

# Precomputed parameters of 20 weak single-qubit unitaries used by the
# ``coherent1e-3_even_depol1e-3_odd_depol5e-4`` preset. Reproduces the exact
# numbers used by the original sabrina workflow.
_COHERENT_1E_3_PARAMS = np.array(
    [
        [3.00547398e-02, 7.77730907e-05, 6.17814209e00],
        [2.76584025e-02, 6.30579886e-04, 5.81529206e00],
        [1.88957563e-02, 3.03815219e-03, 5.74209201e00],
        [1.70521901e-02, 2.14975744e-03, 4.46061858e00],
        [2.85241333e-02, 4.85965411e-03, 4.95735706e00],
        [1.48751794e-02, 9.23733394e-04, 2.75367387e00],
        [2.94347273e-02, 4.10139006e-03, 6.18168685e00],
        [1.92935976e-02, 5.21046056e-03, 2.40779110e00],
        [2.52159703e-02, 2.14070019e-03, 3.03112384e00],
        [2.54789185e-02, 7.76826973e-04, 3.81226679e00],
        [1.04998210e-02, 4.68402509e-03, 1.15964766e00],
        [2.28560583e-02, 1.51864557e-03, 4.86718301e00],
        [2.54664840e-02, 2.08769462e-03, 3.94273381e00],
        [2.47447250e-02, 1.08457883e-03, 2.52471461e00],
        [1.77580516e-02, 3.51123833e-03, 4.68656960e00],
        [1.74055805e-02, 4.83659666e-03, 1.91857332e00],
        [2.70344347e-02, 1.94396868e-04, 4.03463346e00],
        [3.52914278e-03, 7.51015669e-04, 4.22965886e00],
        [3.06779071e-02, 4.50731593e-03, 2.83864261e00],
        [1.11295405e-02, 6.20680458e-03, 3.21719165e00],
    ]
)

# Filenames of SPL coefficient files shipped under examples/data/ and the
# total number of qubits each was generated for.
_SPL_PRESETS = {
    "noise_model_coef_ibm_hanoi_10qchain25_dd_inst1_layer0_unit_100_100s": 10,
    "ext_gen_dict_bigger_231106_nazca_50q": 50,
    "gen_dict_bigger_231127_kyiv_50q": 50,
}


def select_noise_channel(
    noise_str: str, num_qubits: int, tensor_type: str = "mpo"
) -> qtn.MatrixProductOperator | LPDOSuperOperator1D | None:
    """Build a named noise channel as an MPO or an LPDO.

    Args:
        noise_str: identifier from the preset list (e.g. ``"identity"``,
            ``"even_depol1_odd_depol05"``, ``"coherent1e-3_even_depol1e-3_odd_depol5e-4"``,
            or one of the SPL file presets).
        num_qubits: chain length to build the channel on.
        tensor_type: ``"mpo"`` or ``"lpdo"``.

    Returns:
        The constructed channel — an `MatrixProductOperator` if
        ``tensor_type == "mpo"``, an `LPDOSuperOperator1D` if ``"lpdo"``, or
        ``None`` for ``noise_str == "identity"``.
    """
    if noise_str == "identity":
        return None

    if noise_str == "even_depol1_odd_depol05":
        return _layered_depol(0.1, 0.05, num_qubits, tensor_type)

    if noise_str == "even_depol1e-3_odd_depol5e-4":
        return _layered_depol(1e-3, 5e-4, num_qubits, tensor_type)

    if noise_str == "coherent1e-3_even_depol1e-3_odd_depol5e-4":
        params = _COHERENT_1E_3_PARAMS[:num_qubits]
        if tensor_type == "mpo":
            noise_mpo_1 = mpo_channels.single_qubit_coherent_error(params)
            noise_mpo_2 = mpo_channels.dep2loc_mpo_even(1e-3, num_qubits)
            noise_mpo_3 = mpo_channels.dep2loc_mpo_odd(5e-4, num_qubits)
            return noise_mpo_3.apply(noise_mpo_2.apply(noise_mpo_1))
        if tensor_type == "lpdo":
            noise_lpdo_1 = lpdo_channels.single_qubit_coherent_errors(params)
            noise_lpdo_2 = lpdo_channels.dep2loc_even(1e-3, num_qubits)
            noise_lpdo_3 = lpdo_channels.dep2loc_odd(5e-4, num_qubits)
            return compose_lpdos(noise_lpdo_3, compose_lpdos(noise_lpdo_2, noise_lpdo_1))
        raise TypeError(f"Tensor type {tensor_type} not implemented.")

    if noise_str in _SPL_PRESETS:
        max_q = _SPL_PRESETS[noise_str]
        filename = noise_str + ".txt"
        if tensor_type == "mpo":
            return mpo_channels.sparse_pauli_lindblad_mpo_from_file(filename, num_qubits, max_q)
        if tensor_type == "lpdo":
            return lpdo_channels.sparse_pauli_lindblad_lpdo_from_file(filename, num_qubits, max_q)
        raise TypeError(f"Tensor type {tensor_type} not implemented.")

    raise ValueError(f"Noise {noise_str} not implemented yet.")


def _layered_depol(p_even: float, p_odd: float, num_qubits: int, tensor_type: str):
    """Compose two-local depolarising layers on even and odd pairs."""
    if tensor_type == "mpo":
        noise_mpo_1 = mpo_channels.dep2loc_mpo_even(p_even, num_qubits)
        noise_mpo_2 = mpo_channels.dep2loc_mpo_odd(p_odd, num_qubits)
        return noise_mpo_2.apply(noise_mpo_1)
    if tensor_type == "lpdo":
        noise_lpdo_1 = lpdo_channels.dep2loc_even(p_even, num_qubits)
        noise_lpdo_2 = lpdo_channels.dep2loc_odd(p_odd, num_qubits)
        return compose_lpdos(noise_lpdo_2, noise_lpdo_1)
    raise TypeError(f"Tensor type {tensor_type} not implemented.")
