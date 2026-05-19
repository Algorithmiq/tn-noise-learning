# QPT pipeline

The QPT (Quantum Process Tomography) pipeline turns a circuit + noise model
into a synthetic measurement dataset that can train the LPDO.

## Tensor network layout

```
   STATES   |   UNITARY   |    NOISE    |   POVMs
------------|-------------|-------------|------------
            |             |             |
--a--O--i-- | --i--O--j-- | --j--O--k-- | --k--O--b--
```

Every MPO is in PTM form (4-D physical legs). The outer indices `a` and `b`
parameterise the choice of input state and POVM effect, respectively.

## Build the network

[`build_qpt_mpo`][tn_noise_learning.qpt.simulator.build_qpt_mpo] constructs
the whole TN, given:

* `qc_unitary` — Qiskit circuit for the ideal evolution.
* `noise_mpo` — typically the output of
  [`select_noise_channel`][tn_noise_learning.noise_models.channels.select_noise_channel].
* `input_states` — `"oc_pauli_eigenstates"` (6 states/qubit) or `"sic_states"`
  (4 states/qubit) or an explicit array of density matrices.
* `povms` — same options as `input_states`.

For SPAM-aware simulations (separate prep- and measurement-error MPOs), use
[`build_qpt_tn`][tn_noise_learning.qpt.spam.build_qpt_tn] from
`tn_noise_learning.qpt.spam`.

## Sampling strategies

`tn_noise_learning.qpt.sampling` exposes three measurement strategies:

* `local_preps` / `local_meas` — enumerate length-`locality` settings tiled
  across the qubit chain.
* `local_preps_and_meas` — Cartesian product of preparations and measurement
  bases, repeated to reach the target number of shots.
* The `global` and `global_repeated` strategies are produced inline by
  [`generate_data`][tn_noise_learning.qpt.data_generation.generate_data].

## Sample shots

[`simulate_experiments`][tn_noise_learning.qpt.simulator.simulate_experiments]
samples shots qubit-by-qubit, tracing out unmeasured qubits via a tracing
MPS. This keeps the contraction cost scalable: the algorithm never
constructs the full `6^n` probability distribution.

The output is a dict ``{"Exp. m": {input_state_str: outcome_str}}`` that
plays the role of "raw experimental counts" downstream.

## One-shot generation

For the common case, [`generate_data`][tn_noise_learning.qpt.data_generation.generate_data]
wraps everything (and the module exposes a `python -m
tn_noise_learning.qpt.data_generation` CLI). The result is a JSON-ready dict
that [`save_dataset`][tn_noise_learning.qpt.data_generation.save_dataset] can
persist with a timestamped filename.
