# LPDO super-operators

The **Locally Purified Density Operator** (LPDO) representation stores a
quantum channel via its Kraus tensors plus a dual (complex-conjugate) branch.
This guarantees complete positivity by construction: any LPDO is a
mathematically valid channel.

Per-site diagram:

```
       |    <-- outer_upper_ind   (kn{i})
       O
       |\   <-- inner_upper_ind   (k{i})
        )   <-- K (Kraus dimension)
       |/   <-- inner_lower_ind   (b{i})
       O
       |    <-- outer_lower_ind   (bn{i})
```

The full multi-qubit network has two horizontal "tubes" (upper and lower
branches) joined by the local Kraus dimensions. The class
[`LPDOSuperOperator1D`][tn_noise_learning.tn_core.lpdo_model.LPDOSuperOperator1D]
extends quimb's `TensorNetwork1D` to manage the four index families and the
tagging of the upper/lower halves.

## Building one

* [`build_lpdo`][tn_noise_learning.tn_core.lpdo_utils.build_lpdo] — the most
  common entry point. Supports random, identity, and Qiskit-circuit
  initialisations.
* [`LPDOSuperOperator1D.rand`][tn_noise_learning.tn_core.lpdo_model.LPDOSuperOperator1D.rand]
  — bypass `build_lpdo` for fine-grained control of the raw arrays.
* [`compose_lpdos`][tn_noise_learning.tn_core.lpdo_utils.compose_lpdos] —
  compose two LPDOs into a single channel: ``E = E_outer ∘ E_inner``.

## Operations

* [`lpdo_to_mpo`][tn_noise_learning.tn_core.lpdo_utils.lpdo_to_mpo] — contract
  the Kraus dimension to recover an MPO. With `ptm=True` the result is in PTM
  form (the standard for QPT comparisons).
* [`lpdo_to_superop`][tn_noise_learning.tn_core.lpdo_utils.lpdo_to_superop] —
  contract to a dense super-operator matrix.
* [`trace_lpdo`][tn_noise_learning.tn_core.lpdo_utils.trace_lpdo] — used in
  the trace-preservation penalty during training.

## Key parameters

* **`virtual_bd`** (χ) — bond dimension along the chain. Controls how much
  entanglement the channel can encode.
* **`kraus_bd`** (K) — Kraus dimension per site. Higher K = more expressive,
  but also more parameters to train.
