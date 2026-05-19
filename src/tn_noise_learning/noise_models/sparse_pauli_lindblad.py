"""Sparse Pauli-Lindblad (SPL) coefficient file parsing.

The text files shipped with the original sabrina codebase
(``noise_model_coef_*.txt``, ``ext_gen_dict_*.txt``, ``gen_dict_*.txt``)
encode a sparse Pauli-Lindblad model as a flat list of numbers:

* the first ``3 * max_qubits`` entries are the single-qubit ``(x, y, z)``
  coefficients for each qubit, in order;
* the next ``9 * (num_qubits - 1)`` entries are the nine two-qubit
  coefficients ``(c_xx, …, c_zz)`` for each neighbouring pair.

The file-reading + index-arithmetic logic is shared between the LPDO and
MPO implementations and lives here; the channel-construction tensor work is
done in `lpdo_channels.py` / `mpo_channels.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_spl_coefficients(
    filename: str | Path, num_qubits: int, max_qubits: int
) -> tuple[list[tuple[float, float, float]], list[tuple[float, ...]]]:
    """Parse the flat SPL coefficient table from a sabrina-format ``.txt`` file.

    Args:
        filename: path to the coefficient file.
        num_qubits: number of qubits in the chain we want to model.
        max_qubits: total number of qubits the file was generated for (used to
            offset to the two-qubit block, which starts at ``3 * max_qubits``).

    Returns:
        ``(single_qubit_coeffs, two_qubit_coeffs)``:

        * ``single_qubit_coeffs`` — list of length ``num_qubits`` of
          ``(x, y, z)`` tuples.
        * ``two_qubit_coeffs`` — list of length ``num_qubits - 1`` of
          ``(c_xx, c_xy, c_xz, c_yx, c_yy, c_yz, c_zx, c_zy, c_zz)`` tuples.
    """
    content = Path(filename).read_text()
    numbers = re.findall(r"\d.+", content)

    single_qubit_coeffs = []
    for q in range(num_qubits):
        x = float(numbers[3 * q + 0])
        y = float(numbers[3 * q + 1])
        z = float(numbers[3 * q + 2])
        single_qubit_coeffs.append((x, y, z))

    two_qubit_coeffs = []
    for q_link in range(num_qubits - 1):
        base = 3 * max_qubits + 9 * q_link
        coeffs = tuple(float(numbers[base + i]) for i in range(9))
        two_qubit_coeffs.append(coeffs)

    return single_qubit_coeffs, two_qubit_coeffs


# Hadamard-Walsh transforms used to convert Pauli-Lindblad coefficients into
# Kraus probabilities (single-qubit and two-qubit).
HADAMARD_WALSH_1Q: np.ndarray = 0.25 * np.array(
    [
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ]
)

HADAMARD_WALSH_2Q: np.ndarray = np.kron(HADAMARD_WALSH_1Q, HADAMARD_WALSH_1Q)
