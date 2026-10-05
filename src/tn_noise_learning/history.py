"""Per-epoch training record."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from ._io import read_array, read_attr

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure


@dataclass
class History:
    """Metric curves and optional parameter snapshots recorded by [`Trainer`][tn_noise_learning.trainer.Trainer].

    Args:
        metrics: one list per metric name, one value per epoch.
        params: Kraus tensors after each epoch, filled only when ``TrainingConfig.snapshot_params`` is set.
        best_epoch: 1-based epoch whose parameters were written back to the model.
        num_sites: number of qubits, used to draw the trace-preserving reference ``2**num_sites``.
    """

    metrics: dict[str, list[float]] = field(default_factory=dict)
    params: list[list[np.ndarray]] = field(default_factory=list)
    best_epoch: int | None = None
    num_sites: int | None = None

    @property
    def num_epochs(self) -> int:
        """Number of recorded epochs."""
        return max((len(v) for v in self.metrics.values()), default=0)

    def append(self, values: dict[str, Any], tensors: list[np.ndarray] | None = None) -> None:
        """Record one epoch."""
        for name, value in values.items():
            self.metrics.setdefault(name, []).append(float(value))
        if tensors is not None:
            self.params.append([np.asarray(t) for t in tensors])

    def save(self, path: str | Path) -> None:
        """Write to a zarr store at ``path``."""
        group = zarr.open_group(str(path), mode="w")
        group.attrs.update(best_epoch=self.best_epoch, num_sites=self.num_sites, metric_names=list(self.metrics))
        for name, values in self.metrics.items():
            group.create_array(f"metrics/{name}", shape=(len(values),), dtype="float64")[:] = np.asarray(values)
        for epoch, tensors in enumerate(self.params):
            for site, t in enumerate(tensors):
                group.create_array(f"params/epoch_{epoch}/site_{site}", shape=t.shape, dtype=t.dtype)[:] = t

    @classmethod
    def load(cls, path: str | Path) -> History:
        """Read a history written by [`save`][tn_noise_learning.history.History.save]."""
        group = zarr.open_group(str(path), mode="r")
        metrics = {
            str(name): read_array(group, f"metrics/{name}").tolist() for name in read_attr(group, "metric_names")
        }
        params = []
        if "params" in group:
            params_group = zarr.open_group(str(path), mode="r", path="params")
            for epoch in sorted(params_group.keys(), key=lambda k: int(k.split("_")[1])):
                epoch_group = zarr.open_group(str(path), mode="r", path=f"params/{epoch}")
                sites = sorted(epoch_group.keys(), key=lambda k: int(k.split("_")[1]))
                params.append([read_array(epoch_group, site) for site in sites])
        best = read_attr(group, "best_epoch")
        num_sites = read_attr(group, "num_sites")
        return cls(metrics, params, None if best is None else int(best), None if num_sites is None else int(num_sites))

    def plot(self, path: str | Path | None = None) -> Figure:
        """Plot the metric curves with [`plot_history`][tn_noise_learning.plotting.plot_history]."""
        from .plotting import plot_history  # noqa: PLC0415  (avoids a circular import)

        return plot_history(self, path)
