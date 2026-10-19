# Developer guide

## Layout

```
src/tn_noise_learning/
  topology.py   Topology
  lpdo.py       LPDO, index-name helpers, full_network
  circuits.py   circuit_network, product_state_network, apply_circuit
  channels.py   depolarizing, pauli_channel, two_qubit_depolarizing, PAULI_EIGENSTATES
  dataset.py    NoiseDataset (zarr)
  simulate.py   outcome_distribution, sample_shots
  losses.py     probabilities, negative_log_likelihood, tp_penalty
  trainer.py    TrainingConfig, Trainer, project_trace_preserving
  history.py    History
  plotting.py   plot_history
tests/          test_<module>.py per module, test_end_to_end.py, test_package.py
examples/       minimal_example.py
```

## Conventions

- Google docstrings without types; full type hints; `from __future__ import annotations` in every module.
- Ruff (select = ["ALL"]) and ty run through prek; see [Contributing](contributing.md).
- Pass `seed` arguments and use `np.random.default_rng`; never seed global state.
- Tests mirror `src`; dense references use explicit Kraus sums and `qiskit.quantum_info.Operator`.
- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor!:`).

## Extending

- **Loss term**: a callable `(lpdo_tn, probs) -> scalar` built from JAX / quimb operations, registered in
  `TrainingConfig.extra_loss_terms` with its weight (constant or callable of the step).
- **Metric**: a callable `(model: LPDO) -> float` in `TrainingConfig.metrics`; it runs on numpy after each epoch.
- **Learning rate / optimizer**: any optax schedule or callable in `learning_rate`; any factory
  `schedule -> GradientTransformation` in `optimizer`.
- **Noise channel**: return an `LPDO` whose tensors follow the canonical axis order; place two-site blocks with
  `channels._place` as `two_qubit_depolarizing` does.
- **Topology**: `Topology.from_edges`; nothing else needs to change.

## Release

The version comes from git tags (hatch-vcs). Publish a GitHub release with a new tag `vX.Y.Z` on `main`; the
`deploy.yml` workflow builds the sdist and wheel, checks that the version matches the tag, and uploads to PyPI.
