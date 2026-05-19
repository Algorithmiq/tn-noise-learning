# tn-noise-learning

Tensor-network learning of quantum noise channels (LPDO + MPO + JAX) for
quantum process tomography.

`tn_noise_learning` represents quantum channels as **LPDO** (Locally Purified
Density Operator) or **MPO** (Matrix Product Operator) tensor networks in
**PTM** (Pauli Transfer Matrix) form, simulates QPT experiments by tensor
contraction, and reconstructs the noise from sampled outcomes via JAX
autodiff with a KL-divergence + trace-preservation loss.

The library is the engineered/reusable extract of the research codebase
**SABRINA** (*Sensible Approach to Beat Ridiculously Impractical
Noise-characterization Algorithms*).

## Quick links

* [Getting started](getting-started.md) — install, run a 4-qubit toy example.
* Concepts:
  * [LPDO super-operators](concepts/lpdo.md)
  * [Noise channels](concepts/noise-channels.md)
  * [QPT pipeline](concepts/qpt-pipeline.md)
  * [Training](concepts/training.md)
* [Code reference](reference/) — auto-generated from docstrings.

## Pipeline at a glance

```
Input states (OC / SIC-POVM)  ─┐
Unitary circuit (MPO)          ├─▶ build_qpt_mpo()  ─▶  simulate_experiments()
Noise channel (MPO/LPDO)       ┘       │                        │
POVM measurements ─────────────────────┘              JSON experiment data
                                                               │
                                                       training.dataset
                                                               │
                                                       training.optimize_tn
                                                      (JAX + optax, KL loss)
                                                               │
                                                    Optimised LPDO parameters
```
