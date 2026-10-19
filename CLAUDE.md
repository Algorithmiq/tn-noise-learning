# CLAUDE.md

Guidance for Claude Code and other LLM tooling working in this repository.

## Project

`tn_noise_learning`: learn the noise channel of a quantum circuit layer from measurement shots. The channel is an
LPDO (one Kraus tensor per qubit, bonds along a `Topology`), trained by maximum likelihood with JAX/optax plus a
trace-preservation penalty. Pipeline: `Topology -> LPDO -> NoiseDataset (zarr) -> Trainer -> History`.

## Environment

Python 3.12/3.13, dependencies managed by `uv`.

```bash
just build --all-groups
uv run prek install              # git hooks
just lint                        # every hook on every file
just test                        # ~25 s; the 2x2-lattice recovery test is marked `slow`
uv run ruff check . && uv run ruff format --check . && uv run mypy
just docs
just example
```

## Layout

Flat package, one concern per module (see the table in `README.md`): `topology`, `lpdo`, `circuits`, `channels`,
`dataset`, `simulate`, `losses`, `trainer`, `history`, `plotting`. `_jax.py` enables 64-bit JAX and is imported
first by `__init__.py`; `_io.py` holds typed zarr reads. `tests/test_<module>.py` mirrors `src`, plus
`test_end_to_end.py` (recover a known channel from simulated shots) and `test_package.py`.

## Conventions that matter for correctness

- **Sites** are integers `0..n-1` = dataset qubit order = Qiskit qubit index. Edges are sorted `(i, j)`, `i < j`.
- **Kraus tensor axes**: `(*bonds to neighbours in ascending order, kraus, out, in)`.
- **Index names** (`lpdo.py`): bond `v{i}_{j}` / conjugate `w{i}_{j}`, kraus `kr{j}`, output `kn{j}` / `bn{j}`,
  input `k{j}` / `b{j}`; raw (pre-circuit) state legs `p{j}` / `q{j}`; batch hyper-index `exp`.
  The conjugate half is never stored; `full_network` builds it from the Kraus half.
- **Ket/bra**: `k`/`kn` are ket (row) indices, `b`/`bn` are bra (column) indices. A state tensor is `(exp, k, b)`
  with data ρ; an effect tensor is `(exp, bn, kn)` with data E, so the contraction is `Tr[E Λ(ρ)]`.
- **Effect labels**: `2 * basis + bit`; consecutive effect pairs sum to the identity.
- **Gates**: Qiskit matrices are little-endian; `_schmidt_pair` handles the axis order. Gates on non-edge pairs are
  allowed because circuit bonds (`s*`) and model bonds are different index sets.
- Joining the state and effect networks needs `check_collisions=False` (both carry the inner index `exp`).
- Gradients of complex parameters are conjugated before `optimizer.update`. 64-bit JAX is mandatory.
- `tp_penalty` is **squared**: an unsquared norm has a kink at zero and swamps the data term under Adam.
  The likelihood alone rewards inflating the trace; at weight `w` the trained trace exceeds `2**n` by `~1/(2w)`.
- `LPDO.distance` is the squared normalised Frobenius distance `‖A − B‖_F² / 4**n`.
- Early stopping monitors `test_loss`, never the distance to the ideal channel (that would leak the ground truth).

## Testing policy

- Tiny fixtures: `Topology.chain(2)`, `Topology.square(2, 2)`; dense references are built with explicit Kraus sums
  and `qiskit.quantum_info.Operator`, never with another tensor-network path.
- `pytest.raises(..., match=...)` always. Mark tests over ~10 s as `slow`.
- Do not add notebooks, data files or figures to the repository.

## Do not

- `print` in library code (ruff T20); use `logging`.
- Global RNG seeding; pass `seed` and use `np.random.default_rng`.
- Plotting outside `plotting.py`; matplotlib is an optional extra.
- Re-introduce 1D-only code paths; everything must work on an arbitrary `Topology`.
