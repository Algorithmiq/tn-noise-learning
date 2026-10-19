# tn-noise-learning

Learn the noise channel of a quantum circuit layer from measurement shots, with tensor networks.

A quantum channel on `n` qubits is stored as a **locally purified density operator (LPDO)**: one Kraus tensor
`V_j` per qubit with a Kraus index of size `K`, joined to its neighbours by virtual bonds of size `χ` along a chosen
`Topology`. The channel `Λ(ρ) = Σ V ρ V†` is completely positive by construction. Its parameters are fitted by
maximum likelihood to shots of product-state preparations and product measurements, with a soft penalty that
keeps the channel trace preserving.

```
Topology  ─►  LPDO  ─►  NoiseDataset (zarr)  ─►  Trainer  ─►  History.plot()
```

- [Installation](installation.md): how to install the package.
- [Quickstart](quickstart.md): the example script, step by step.
- [Concepts](concepts.md): the LPDO, the probability network, the loss, how shots are simulated.
- [Developer guide](developer.md): layout, conventions, how to extend.
- [Contributing](contributing.md): development environment, everyday commands, pull requests.
- [References](references.md): the papers this package builds on.
- [Code reference](reference/tn_noise_learning/index.md): generated from the docstrings.
