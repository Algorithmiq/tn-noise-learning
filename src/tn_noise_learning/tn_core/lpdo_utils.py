"""LPDO construction, composition, conversion, and basic measurements.

These helpers turn raw arrays or Qiskit circuits into an
[`LPDOSuperOperator1D`][tn_noise_learning.tn_core.lpdo_model.LPDOSuperOperator1D]
and provide the standard transformations needed by the QPT pipeline (channel
composition, conversion to MPO/superoperator, trace, fidelity).
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain

import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix

from .lpdo_model import LPDOSuperOperator1D
from .mpo_utils import qc_to_mpo


def random_complex_gaussian(
    shape: tuple[int, ...],
    mu: float = 0,
    sigma: float = 1 / np.sqrt(2),
    num_qubits: int | None = None,
    virtual_bd: int | None = None,
    kraus_bd: int | None = None,
) -> np.ndarray:
    """Draw a complex Gaussian tensor, optionally rescaled so the LPDO is normalised in expectation."""
    re = mu + np.random.randn(*shape) * sigma
    im = mu + np.random.randn(*shape) * sigma
    z = re + 1j * im

    if (num_qubits is not None) and (kraus_bd is not None) and (virtual_bd is not None):
        alpha = 8 * sigma**2
        fact = np.sqrt(2 / (alpha * kraus_bd * virtual_bd ** (1.0 - 1.0 / num_qubits)))
        z = z * fact
    return z


def random_complex_uniform(shape: tuple[int, ...], mu: float = 0, sigma: float = 0.1) -> np.ndarray:
    """Draw a complex tensor with entries uniform in ``[-sigma/2, sigma/2]``."""
    re = sigma * (np.random.rand(*shape) - 0.5)
    im = sigma * (np.random.rand(*shape) - 0.5)
    return re + 1j * im


def build_lpdo(
    num_qubits: int = 2,
    kraus_bd: int = 1,
    virtual_bd: int = 1,
    phys_dim: int = 2,
    tn_data: Iterable | QuantumCircuit | None = None,
    init_method: str = "normal_complex",
    dtype: type[complex] = complex,
    seed: int | None = None,
) -> LPDOSuperOperator1D:
    """Construct an [`LPDOSuperOperator1D`][tn_noise_learning.tn_core.lpdo_model.LPDOSuperOperator1D].

    Three input modes:
        * ``tn_data is None`` — random or identity initialisation governed by ``init_method``.
        * ``tn_data: QuantumCircuit`` — initialise from the unitary MPO of a Qiskit circuit.
        * ``tn_data: Iterable`` — explicit sequence of site tensors.

    Args:
        num_qubits: number of qubits in the chain.
        kraus_bd: Kraus dimension K.
        virtual_bd: virtual bond dimension χ.
        phys_dim: physical dimension per site (2 for qubits).
        tn_data: see above.
        init_method: ``"normal_complex"``, ``"uniform_complex"``, or ``"identity"``.
        dtype: numpy dtype for the tensors.
        seed: optional RNG seed for reproducibility.
    """
    if isinstance(seed, int):
        np.random.seed(seed)

    end_shape = (virtual_bd, kraus_bd, phys_dim, phys_dim)
    mid_shape = (virtual_bd, virtual_bd, kraus_bd, phys_dim, phys_dim)

    if tn_data is None:
        if init_method == "normal_complex":
            sigma = 1 / np.sqrt(2)
            start_tensor_data = random_complex_gaussian(
                end_shape, sigma=sigma, num_qubits=num_qubits, virtual_bd=virtual_bd, kraus_bd=kraus_bd
            )
            mid_tensors_data = [
                random_complex_gaussian(
                    mid_shape, sigma=sigma, num_qubits=num_qubits, virtual_bd=virtual_bd, kraus_bd=kraus_bd
                )
                for _ in range(num_qubits - 2)
            ]
            end_tensor_data = random_complex_gaussian(
                end_shape, sigma=sigma, num_qubits=num_qubits, virtual_bd=virtual_bd, kraus_bd=kraus_bd
            )
        elif init_method == "uniform_complex":
            start_tensor_data = random_complex_uniform(end_shape)
            mid_tensors_data = [random_complex_uniform(mid_shape) for _ in range(num_qubits - 2)]
            end_tensor_data = random_complex_uniform(end_shape)
        elif init_method == "identity":
            start_tensor_data = np.array([[np.eye(2)]], dtype=dtype)
            mid_tensors_data = np.array([[[[np.eye(2)]]] for _ in range(num_qubits - 2)], dtype=dtype)
            end_tensor_data = np.array([[np.eye(2)]], dtype=dtype)
        else:
            raise ValueError(f"{init_method} not implemented.")

    elif isinstance(tn_data, QuantumCircuit):
        lpdo = qc_to_mpo(tn_data)
        for k, t in enumerate(lpdo.tensors):
            t.new_ind(f"kraus{k}", size=1, axis=-3)
        tn_data = [t.data.astype(dtype) for t in lpdo.tensors]
        start_tensor_data, *mid_tensors_data, end_tensor_data = tn_data

    elif isinstance(tn_data, Iterable):
        start_tensor_data, *mid_tensors_data, end_tensor_data = tn_data
    else:
        raise ValueError(f"Unable to interpret tn_data {type(tn_data)} to build an LPDO.")

    # Interleave each Kraus tensor with its complex conjugate (the LPDO dual branch).
    ti = [start_tensor_data, np.conj(start_tensor_data)]
    tm = list(chain.from_iterable([[tn, np.conj(tn)] for tn in mid_tensors_data]))
    tl = [end_tensor_data, np.conj(end_tensor_data)]
    tensor_data = ti + tm + tl

    return LPDOSuperOperator1D(
        tensor_data,
        outer_upper_ind_id="kn{}",
        inner_upper_ind_id="k{}",
        inner_lower_ind_id="b{}",
        outer_lower_ind_id="bn{}",
        site_tag_id="I{}",
        tags="LPDO",
    )


def compose_lpdos(lpdo_outer: LPDOSuperOperator1D, lpdo_inner: LPDOSuperOperator1D) -> LPDOSuperOperator1D:
    r"""Compose two LPDOs: ``E = E_outer ∘ E_inner``.

    Conceptually ``lpdo_inner`` acts first, then ``lpdo_outer``. The two
    networks are joined along their inner physical bonds.
    """
    num_qubits = lpdo_outer.L
    kraus_1 = lpdo_outer.select(["V"]).copy()
    kraus_2 = lpdo_inner.select(["V"]).copy()

    phys_inds = [f"kn{q}" for q in range(num_qubits)] + [f"k{q}" for q in range(num_qubits)]

    kraus_inds_1 = [ind for ind in list(kraus_1.outer_inds()) if ind not in phys_inds]
    kraus_inds_2 = [ind for ind in list(kraus_2.outer_inds()) if ind not in phys_inds]

    # Reindex Kraus legs to fresh unique names to avoid collisions when the
    # same LPDO is composed with itself (otherwise both copies would share
    # Kraus index names and quimb would treat them as a single bond).
    rename_1 = {ind: qtn.rand_uuid() for ind in kraus_inds_1}
    rename_2 = {ind: qtn.rand_uuid() for ind in kraus_inds_2}
    kraus_1.reindex_(rename_1)
    kraus_2.reindex_(rename_2)
    kraus_inds_1 = [rename_1[ind] for ind in kraus_inds_1]
    kraus_inds_2 = [rename_2[ind] for ind in kraus_inds_2]

    kraus_pairs = list(zip(kraus_inds_1, kraus_inds_2, strict=False))

    tmp = "k_temp{}"
    kraus_1.reindex_({kraus_1._inner_upper_ind_id.format(q): tmp.format(q) for q in range(num_qubits)})
    kraus_2.reindex_({kraus_2._outer_upper_ind_id.format(q): tmp.format(q) for q in range(num_qubits)})
    combined_kraus = qtn.TensorNetwork([kraus_1, kraus_2])

    for q in range(num_qubits):
        combined_kraus.contract_ind(f"k_temp{q}")
    combined_kraus.fuse_multibonds_()

    for qub, tn in enumerate(combined_kraus.tensors):
        tn.fuse_({f"kr{qub}": kraus_pairs[qub]})

    combined_lpdo = build_lpdo(tn_data=[t.data for t in combined_kraus.tensors])
    combined_lpdo.mangle_inner_()
    return combined_lpdo


def lpdo_to_mpo(
    self: LPDOSuperOperator1D, upper_ind_id: str = "k{}", lower_ind_id: str = "b{}", ptm: bool = True
) -> qtn.MatrixProductOperator:
    """Contract the Kraus dimension to convert an LPDO into an MPO super-operator.

    Args:
        self: input LPDO.
        upper_ind_id: format string for the resulting MPO upper indices.
        lower_ind_id: format string for the resulting MPO lower indices.
        ptm: if True (default), apply the change of basis from computational
            to Pauli (PTM) form on each site.
    """
    tn = self.copy()
    num_qubits = tn.L

    kraus_bonds_names = set(tn.select(["V"]).outer_inds()).difference(tn.outer_inds())
    for bd in kraus_bonds_names:
        tn.contract_ind(bd)
    tn.fuse_multibonds_()

    for tag in tn.site_tags:
        n = tag[1:]
        tn[tag].fuse_(
            {
                upper_ind_id.format(n): (tn.outer_lower_ind_id.format(n), tn.outer_upper_ind_id.format(n)),
                lower_ind_id.format(n): (tn.inner_lower_ind_id.format(n), tn.inner_upper_ind_id.format(n)),
            }
        )

    mpo = tn.view_as(
        qtn.MatrixProductOperator,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id="I{}",
        cyclic=False,
        L=tn.L,
    )

    if not ptm:
        return mpo

    comp_basis_to_ptm = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, -1j, +1j, 0],
            [1, 0, 0, -1],
        ]
    ) / np.sqrt(2)
    ptm_to_comp_basis = np.conj(comp_basis_to_ptm).T

    tmp_upper_ind_id = "up{}"
    tmp_lower_ind_id = "lo{}"
    mpo.reindex_upper_sites_(tmp_upper_ind_id)
    mpo.reindex_lower_sites_(tmp_lower_ind_id)
    mpo_ptm = qtn.TensorNetwork(
        [
            qtn.MPO_product_operator(
                [comp_basis_to_ptm] * num_qubits, upper_ind_id=upper_ind_id, lower_ind_id=tmp_upper_ind_id
            ),
            qtn.MPO_product_operator(
                [ptm_to_comp_basis] * num_qubits, upper_ind_id=tmp_lower_ind_id, lower_ind_id=lower_ind_id
            ),
            mpo,
        ],
        virtual=False,
    )

    for tag in mpo.site_tags:
        mpo_ptm.contract_tags_(tag)
    mpo_ptm.fuse_multibonds_()

    return mpo_ptm.view_as(
        qtn.MatrixProductOperator,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        site_tag_id="I{}",
        cyclic=False,
        L=tn.L,
    )


def lpdo_to_superop(lpdo: LPDOSuperOperator1D) -> np.ndarray:
    """Contract an LPDO to a dense super-operator matrix."""
    num_qubits = lpdo.L
    left_inds = [lpdo.lower_outer_ind_id.format(_) for _ in range(num_qubits)] + [
        lpdo.upper_outer_ind_id.format(_) for _ in range(num_qubits)
    ]
    right_inds = [lpdo.lower_inner_ind_id.format(_) for _ in range(num_qubits)] + [
        lpdo.upper_inner_ind_id.format(_) for _ in range(num_qubits)
    ]
    return lpdo.contract().to_dense(left_inds, right_inds)


def lpdo_fidelity(lpdo_a: LPDOSuperOperator1D, lpdo_b: LPDOSuperOperator1D, backend: str = "jax") -> float:
    """Squared normalised tensor-network distance between two LPDOs."""
    num_qubits = lpdo_a.L
    fid = qtn.tensor_network_distance(lpdo_a, lpdo_b, backend=backend) / 2**num_qubits
    return fid**2


def state_mpo(tensor_data: np.ndarray, output_ind: str = "k{}", input_ind: str = "b{}") -> qtn.MatrixProductOperator:
    r"""MPO encoding of an input state on each qubit.

    Per-site shape::

        k - -O - -b
    """
    return qtn.MPO_product_operator(tensor_data, upper_ind_id=output_ind, lower_ind_id=input_ind, tags="IN")


def povm_mpo(tensor_data: np.ndarray, output_ind: str = "kn{}", input_ind: str = "bn{}") -> qtn.MatrixProductOperator:
    r"""MPO encoding of a POVM effect on each qubit.

    Per-site shape::

        bn - -O - -kn

    The input tensors are transposed on their last two axes to match the
    contraction convention used when combining with an LPDO.
    """
    return qtn.MPO_product_operator(
        np.swapaxes(tensor_data, -1, -2), upper_ind_id=output_ind, lower_ind_id=input_ind, tags="OUT"
    )


def process_tn(
    lpdo: LPDOSuperOperator1D, state_label: Iterable[int], povm_label: Iterable[int]
) -> tuple[qtn.TensorNetwork, qtn.MatrixProductOperator, qtn.MatrixProductOperator]:
    r"""Tensor network for the QPT transition probability of a single ``(state, POVM)`` pair.

    Network shape::

        |  _____kraus_index___|_
        |/                    |/
    kn--O--k  k--STATE--b  b--O^dag--bn  bn--POVM--kn
        | ___      |       ___|_             |
        :/                    :/

    Contracting all external (kn) bonds yields
    ``Tr[LPDO[ρ_α] POVM_β] = P(POVM_β | ρ_α)``.
    """
    # Lazy import avoids a circular dependency: qpt depends on tn_core but
    # this helper happens to need the Pauli eigenstate basis defined in qpt.
    from tn_noise_learning.qpt.states import pauli_eigenstates

    oc_pauli_ops = pauli_eigenstates()

    def op_selector(label):
        return np.array([oc_pauli_ops[_] for _ in label])

    state = state_mpo(op_selector(state_label))
    povm = povm_mpo(op_selector(povm_label))

    return qtn.TensorNetwork([lpdo, state, povm]), state, povm


def transition_prob_qiskit(qc: QuantumCircuit, state_label: Iterable[int], povm_label: Iterable[int]) -> complex:
    r"""Compute the QPT transition probability using Qiskit's dense simulator.

    ``P(POVM_β | U[ρ_α]) = Tr[POVM_β · U[ρ_α]]`` — useful for unit-testing the
    tensor-network pipeline against a small dense reference.
    """
    if qc.num_qubits > 10:
        print(f"Number of qubits = {qc.num_qubits}. This may take a while to compute.")  # noqa: T201

    pauli_int_to_string = {"0": "0", "1": "1", "2": "+", "3": "-", "4": "r", "5": "l"}
    state = DensityMatrix.from_label("".join([pauli_int_to_string[str(a)] for a in state_label[::-1]]))
    state = state.evolve(qc).data
    povm = DensityMatrix.from_label("".join([pauli_int_to_string[str(b)] for b in povm_label[::-1]])).data
    return np.trace(povm @ state)


def trace_lpdo(self: LPDOSuperOperator1D):
    r"""Trace an LPDO by connecting matching inner (k, b) and outer (kn, bn) bonds.

    Used as a building block for the trace-preservation penalty in training.
    """
    tn = self.copy()
    inner_connect = {tn.inner_upper_ind_id.format(n): tn.inner_lower_ind_id.format(n) for n in range(tn.L)}
    outer_connect = {tn.outer_upper_ind_id.format(n): tn.outer_lower_ind_id.format(n) for n in range(tn.L)}
    connect_bonds = inner_connect | outer_connect
    return tn.reindex(connect_bonds).contract(backend="jax")
