# Examples

`minimal_example.py` runs the whole pipeline on a 2x2 qubit lattice: simulate shots from a known two-qubit
depolarizing channel behind a layer of CZ gates, save them as a zarr dataset, train a random LPDO on them, and plot
the training curves. It takes about a minute on a laptop.

```bash
uv sync --all-extras
uv run python examples/minimal_example.py
```

To use your own data, write a [`NoiseDataset`](../src/tn_noise_learning/dataset.py) with your shot labels, the
single-qubit states and effects they refer to, and (optionally) the ideal circuit, then call `Trainer.fit` as above.
