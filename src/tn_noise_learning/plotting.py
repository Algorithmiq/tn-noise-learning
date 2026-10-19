# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training-curve figure; requires the ``plot`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from .history import History

LOG_SCALE = {"tp_penalty", "distance_to_ideal"}
STANDARD_PANELS = ("loss", "nll", "tp_penalty", "trace", "distance_to_ideal")


def plot_history(history: History, path: str | Path | None = None) -> Figure:
    """One panel per metric: loss (train and test), NLL, TP penalty, trace, distance to ideal, then custom metrics."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415  (optional dependency)
    except ImportError as error:
        raise ImportError(
            "Plotting requires matplotlib; install the extra with `pip install tn-noise-learning[plot]`."
        ) from error

    metrics = history.metrics
    custom = [name for name in metrics if name not in {*STANDARD_PANELS, "train_loss", "test_loss"}]
    panels = [p for p in STANDARD_PANELS if p == "loss" or p in metrics] + custom
    epochs = range(1, history.num_epochs + 1)

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2), squeeze=False)
    for ax, name in zip(axes[0], panels, strict=True):
        if name == "loss":
            ax.plot(epochs, metrics["train_loss"], label="train")
            if "test_loss" in metrics:
                ax.plot(epochs, metrics["test_loss"], label="test")
            ax.legend()
        else:
            ax.plot(epochs, metrics[name])
        if name == "trace" and history.num_sites is not None:
            ax.axhline(2**history.num_sites, color="gray", linestyle="--", linewidth=1)
        if name in LOG_SCALE:
            ax.set_yscale("log")
        if history.best_epoch is not None:
            ax.axvline(history.best_epoch, color="gray", linestyle=":", linewidth=1)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("epoch")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
