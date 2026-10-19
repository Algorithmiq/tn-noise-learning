# Concepts

## LPDO

Site `j` holds a Kraus tensor `V_j` with axes `(*bonds, kraus, out, in)`. The full network joins `V_j` with its
complex conjugate through the shared Kraus index, so the channel is `Λ(ρ) = Σ_kr V ρ V†` and completely positive
for any parameter values. Bonds `v{i}_{j}` (Kraus half) and `w{i}_{j}` (conjugate half) follow the topology edges.

```
      kn0      kn1      kn2          <- output, ket side
       |        |        |
   [V_0]==v0_1==[V_1]==v1_2==[V_2]    kraus kr_j shared vertically
       |\       |\       |\
       k0 kr0   k1 kr1   k2 kr2      <- input, ket side
       b0 |     b1 |     b2 |        <- input, bra side
       |/       |/       |/
   [V_0*]=w0_1=[V_1*]=w1_2=[V_2*]
       |        |        |
      bn0      bn1      bn2          <- output, bra side
```

[`LPDO.random`][tn_noise_learning.lpdo.LPDO.random] scales complex Gaussian tensors so that the expected trace
is `2**n`; [`LPDO.trace`][tn_noise_learning.lpdo.LPDO.trace] equals `2**n` for a trace-preserving channel.
[`LPDO.distance`][tn_noise_learning.lpdo.LPDO.distance] is `‖Λ − Λ'‖_F² / 4**n`.

## Probability network

For a batch of shots the prepared single-qubit states form tensors `(exp, k_j, b_j)` and the measured effects
tensors `(exp, bn_j, kn_j)`, all sharing the batch hyper-index `exp`. The ideal circuit sits between the states
and the channel as one tensor per site: every two-qubit gate is split by an operator-Schmidt decomposition into two
site tensors joined by a bond, so the construction works on any connectivity without compression. Contracting
everything with `exp` left open gives all `Tr[E Λ(U ρ U†)]` of the batch in one cotengra-optimised contraction.

## Loss

```
L(θ) = −mean log p_i(θ)  +  w · ‖Tr_out Λ(θ) − 𝟙‖_F² / 2**n
```

The first term is the negative log-likelihood of the shots. The second keeps the channel trace preserving; it is
squared so that its gradient vanishes on the manifold and large weights act as a smooth constraint. Because the
likelihood alone can be lowered by inflating the channel's trace, the trained trace sits at `2**n (1 + 1/(2w))`;
the default `w = 100` leaves a 0.5 % bias. Custom terms with the same `(lpdo_tn, probs) -> scalar` signature can
be added through `TrainingConfig.extra_loss_terms`.

## Simulated shots

[`sample_shots`][tn_noise_learning.simulate.sample_shots] draws random `(preparation, basis)` settings, contracts
the exact joint distribution over the `2**n` outcomes of each setting once, and draws all its shots from a single
multinomial. The default preparations and effects are the six Pauli eigenstates; effect `2 * basis + bit` is the
projector onto outcome `bit` of basis Z, X or Y.
