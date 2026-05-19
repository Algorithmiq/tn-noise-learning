# Noise channels

`tn_noise_learning.noise_models` provides parallel MPO and LPDO constructors
for the noise channels used by the QPT pipeline.

## Dispatcher

The simplest entry point is
[`select_noise_channel`][tn_noise_learning.noise_models.channels.select_noise_channel]:

```python
from tn_noise_learning import select_noise_channel

# Two-local depolarising on even/odd pairs.
noise = select_noise_channel("even_depol1e-3_odd_depol5e-4", num_qubits=10, tensor_type="mpo")

# Sparse Pauli-Lindblad from a coefficient file.
noise = select_noise_channel("noise_model_coef_ibm_hanoi_10qchain25_dd_inst1_layer0_unit_100_100s",
                             num_qubits=10, tensor_type="lpdo")
```

The string presets are:

* `identity` — returns `None` (no noise).
* `even_depol1_odd_depol05` — two-local depolarising at `p = 0.1 / 0.05`.
* `even_depol1e-3_odd_depol5e-4` — same shape, weaker rates.
* `coherent1e-3_even_depol1e-3_odd_depol5e-4` — coherent unitary errors
  (precomputed) composed with the two-local layers.
* SPL coefficient files (must live in `examples/data/`):
  * `noise_model_coef_ibm_hanoi_10qchain25_dd_inst1_layer0_unit_100_100s` (10 qubits, IBM Hanoi)
  * `ext_gen_dict_bigger_231106_nazca_50q` (50 qubits, Nazca)
  * `gen_dict_bigger_231127_kyiv_50q` (50 qubits, Kyiv)

## Lower-level constructors

### MPO form (PTM, 4-D physical legs)

* [`dephasing_ptm`][tn_noise_learning.noise_models.mpo_channels.dephasing_ptm],
  [`depolarizing_ptm`][tn_noise_learning.noise_models.mpo_channels.depolarizing_ptm],
  [`amplitude_damping_ptm`][tn_noise_learning.noise_models.mpo_channels.amplitude_damping_ptm],
  [`pauli_channel_ptm`][tn_noise_learning.noise_models.mpo_channels.pauli_channel_ptm]
  — single-qubit PTMs.
* [`dep2loc_mpo_even`][tn_noise_learning.noise_models.mpo_channels.dep2loc_mpo_even],
  [`dep2loc_mpo_odd`][tn_noise_learning.noise_models.mpo_channels.dep2loc_mpo_odd]
  — two-local depolarising on even/odd pairs.
* [`sparse_pauli_lindblad_mpo_from_file`][tn_noise_learning.noise_models.mpo_channels.sparse_pauli_lindblad_mpo_from_file].

### LPDO form (Kraus, 2-D physical legs)

Parallels to the MPO constructors:

* [`single_qubit_coherent_errors`][tn_noise_learning.noise_models.lpdo_channels.single_qubit_coherent_errors]
* [`two_qubit_depolarizing_error`][tn_noise_learning.noise_models.lpdo_channels.two_qubit_depolarizing_error]
* [`dep2loc_even`][tn_noise_learning.noise_models.lpdo_channels.dep2loc_even],
  [`dep2loc_odd`][tn_noise_learning.noise_models.lpdo_channels.dep2loc_odd]
* [`sparse_pauli_lindblad_lpdo_from_file`][tn_noise_learning.noise_models.lpdo_channels.sparse_pauli_lindblad_lpdo_from_file]

## SPL file format

The `.txt` files in `examples/data/` encode a sparse Pauli-Lindblad model as a
flat list of numbers parsed by
[`parse_spl_coefficients`][tn_noise_learning.noise_models.sparse_pauli_lindblad.parse_spl_coefficients]:

* The first `3 × max_qubits` entries hold the single-qubit `(x, y, z)` coefficients per qubit.
* The next `9 × (num_qubits − 1)` entries hold the two-qubit `(c_xx, …, c_zz)` coefficients per neighbouring pair.
