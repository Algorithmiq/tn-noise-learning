# Testing

- `tests/test_<module>.py` mirrors `src/tn_noise_learning/<module>.py`; `test_end_to_end.py` recovers a known channel
  from simulated shots.
- Fixtures are tiny: `Topology.chain(2)` and `Topology.square(2, 2)`. Dense references are built with explicit Kraus
  sums and `qiskit.quantum_info.Operator`, never with another tensor-network path.
- Use `pytest.raises(..., match=...)`. Mark tests that take more than about 10 s with `@pytest.mark.slow`.
- Warnings are errors. A third-party warning that cannot be fixed gets a narrowly scoped `ignore` in
  `[tool.pytest.ini_options]`, with a comment saying why.
- Pass a `seed` and use `np.random.default_rng`; never seed global state.
- CI runs the suite on Linux x86-64, Linux ARM64 and macOS with Python 3.12 and 3.15 (`.github/workflows/test.yml`),
  and once with coverage (`qa-analysis.yml`).
