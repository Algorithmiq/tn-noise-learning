"""Shot-level noise-learning dataset stored as a zarr group.

Layout of ``dataset.zarr``::

    attrs      format_version, num_qubits, created, circuit_qasm2 (optional)
    inputs     int8        (num_shots, num_qubits)   label of the prepared state on each qubit, index into states[j]
    outcomes   int8        (num_shots, num_qubits)   label of the measured effect on each qubit, index into povms[j]
    states     complex128  (num_qubits, num_states, 2, 2)
    povms      complex128  (num_qubits, num_effects, 2, 2)
    ideal_channel/   optional LPDO group (see LPDO.to_group) used only to report the distance during training
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import zarr
from qiskit import QuantumCircuit, qasm2

from .lpdo import LPDO

FORMAT_VERSION = 1
ARRAY_NAMES = ("inputs", "outcomes", "states", "povms")


class Batch(NamedTuple):
    """One training batch: per-site state and effect matrices, each of shape ``(num_qubits, batch, 2, 2)``."""

    states: np.ndarray
    povms: np.ndarray


@dataclass
class NoiseDataset:
    """Shots of a process-tomography experiment on product states with product measurements.

    Args:
        inputs: prepared-state labels, shape ``(num_shots, num_qubits)``.
        outcomes: measured-effect labels, shape ``(num_shots, num_qubits)``.
        states: single-qubit input states per qubit, shape ``(num_qubits, num_states, 2, 2)``.
        povms: single-qubit effects per qubit, shape ``(num_qubits, num_effects, 2, 2)``.
        circuit: ideal circuit applied before the noise channel, or ``None`` for the identity.
        ideal_channel: ground-truth channel, if known, to report the distance to during training.
    """

    inputs: np.ndarray
    outcomes: np.ndarray
    states: np.ndarray
    povms: np.ndarray
    circuit: QuantumCircuit | None = None
    ideal_channel: LPDO | None = None

    def __post_init__(self) -> None:
        """Normalise dtypes and validate shapes and label ranges."""
        self.inputs = np.asarray(self.inputs, dtype=np.int8)
        self.outcomes = np.asarray(self.outcomes, dtype=np.int8)
        self.states = np.asarray(self.states, dtype=complex)
        self.povms = np.asarray(self.povms, dtype=complex)
        n = self.num_qubits
        if self.inputs.shape != self.outcomes.shape or self.inputs.ndim != 2:
            raise ValueError("inputs and outcomes must both have shape (num_shots, num_qubits).")
        if self.states.shape[0] != n or self.states.shape[-2:] != (2, 2) or self.povms.shape[0] != n or self.povms.shape[-2:] != (2, 2):
            raise ValueError("states and povms must have shape (num_qubits, num_labels, 2, 2).")
        if self.inputs.min(initial=0) < 0 or self.inputs.max(initial=0) >= self.states.shape[1]:
            raise ValueError("inputs contain labels outside the range of states.")
        if self.outcomes.min(initial=0) < 0 or self.outcomes.max(initial=0) >= self.povms.shape[1]:
            raise ValueError("outcomes contain labels outside the range of povms.")
        if self.circuit is not None and self.circuit.num_qubits != n:
            raise ValueError(f"Circuit acts on {self.circuit.num_qubits} qubits, dataset has {n}.")
        if self.ideal_channel is not None and self.ideal_channel.num_sites != n:
            raise ValueError(f"Ideal channel has {self.ideal_channel.num_sites} sites, dataset has {n} qubits.")

    @property
    def num_shots(self) -> int:
        """Number of recorded shots."""
        return self.inputs.shape[0]

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return self.inputs.shape[1]

    def __len__(self) -> int:
        """Number of recorded shots."""
        return self.num_shots

    def subset(self, indices: np.ndarray) -> NoiseDataset:
        """Dataset restricted to the given shots; states, povms, circuit and ideal channel are shared."""
        return replace(self, inputs=self.inputs[indices], outcomes=self.outcomes[indices])

    def split(self, test_fraction: float = 0.1, seed: int = 0) -> tuple[NoiseDataset, NoiseDataset]:
        """Random disjoint ``(train, test)`` split."""
        if not 0 < test_fraction < 1:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}.")
        permutation = np.random.default_rng(seed).permutation(self.num_shots)
        num_test = max(1, round(test_fraction * self.num_shots))
        return self.subset(permutation[num_test:]), self.subset(permutation[:num_test])

    def gather(self, indices: np.ndarray) -> Batch:
        """State and effect matrices of the given shots, shape ``(num_qubits, len(indices), 2, 2)`` each."""
        sites = np.arange(self.num_qubits)[:, None]
        return Batch(self.states[sites, self.inputs[indices].T], self.povms[sites, self.outcomes[indices].T])

    def batches(self, batch_size: int, *, shuffle: bool = True, seed: int = 0, drop_last: bool = True) -> Iterator[Batch]:
        """Iterate over batches; with ``drop_last`` every batch has exactly ``batch_size`` shots (one JIT trace)."""
        if batch_size > self.num_shots:
            raise ValueError(f"batch_size {batch_size} exceeds the {self.num_shots} shots in the dataset.")
        order = np.random.default_rng(seed).permutation(self.num_shots) if shuffle else np.arange(self.num_shots)
        stop = self.num_shots - (self.num_shots % batch_size if drop_last else 0)
        for start in range(0, stop, batch_size):
            yield self.gather(order[start : start + batch_size])

    def save(self, path: str | Path) -> None:
        """Write the dataset to a zarr store at ``path`` (overwriting)."""
        group = zarr.open_group(str(path), mode="w")
        group.attrs.update(format_version=FORMAT_VERSION, num_qubits=self.num_qubits, created=datetime.now(UTC).isoformat())
        if self.circuit is not None:
            group.attrs["circuit_qasm2"] = qasm2.dumps(self.circuit)
        for name in ARRAY_NAMES:
            array = getattr(self, name)
            chunks = (min(self.num_shots, 65536), self.num_qubits) if name in ("inputs", "outcomes") else array.shape
            group.create_array(name, shape=array.shape, dtype=array.dtype, chunks=chunks)[:] = array
        if self.ideal_channel is not None:
            self.ideal_channel.to_group(group.create_group("ideal_channel"))

    @classmethod
    def load(cls, path: str | Path) -> NoiseDataset:
        """Read a dataset written by [`save`][tn_noise_learning.dataset.NoiseDataset.save]."""
        group = zarr.open_group(str(path), mode="r")
        arrays = {name: group[name][:] for name in ARRAY_NAMES}  # type: ignore[index]
        circuit = None
        if "circuit_qasm2" in group.attrs:
            circuit = qasm2.loads(str(group.attrs["circuit_qasm2"]), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        ideal = LPDO.from_group(group["ideal_channel"]) if "ideal_channel" in group else None  # type: ignore[arg-type]
        return cls(**arrays, circuit=circuit, ideal_channel=ideal)
