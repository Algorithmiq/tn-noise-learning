# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loss terms for LPDO training: transition probabilities, negative log-likelihood, trace-preservation penalty.

Every loss term has the signature ``term(lpdo_tn, probs) -> scalar`` so that custom terms plug into
[`TrainingConfig.extra_loss_terms`][tn_noise_learning.trainer.TrainingConfig] exactly like the built-in ones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
import quimb.tensor as qtn
from jax import Array

from .circuits import BATCH_IND, product_state_network
from .lpdo import input_ind, output_ind, site_tag

if TYPE_CHECKING:
    import numpy as np

LossTerm = Callable[[qtn.TensorNetwork, jnp.ndarray], jnp.ndarray]

PROBABILITY_FLOOR = 1e-15


def measurement_network(povms: np.ndarray | Array) -> qtn.TensorNetwork:
    """Batched POVM effects ``povms[j, exp]`` (shape ``(num_sites, batch, 2, 2)``) as tensors ``(exp, bn_j, kn_j)``."""
    return qtn.TensorNetwork(
        [
            qtn.Tensor(e, inds=(BATCH_IND, output_ind(j, dagger=True), output_ind(j)), tags=site_tag(j))
            for j, e in enumerate(povms)
        ]
    )


def probabilities(
    lpdo_tn: qtn.TensorNetwork,
    states: np.ndarray | Array,
    povms: np.ndarray | Array,
    circuit_tn: qtn.TensorNetwork | None = None,
    optimize: str = "auto-hq",
) -> jnp.ndarray:
    """Transition probabilities ``Tr[E Λ(U ρ U†)]`` for a batch of ``(state, effect)`` pairs, clipped from below."""
    parts = [lpdo_tn, product_state_network(states, raw=circuit_tn is not None), measurement_network(povms)]
    if circuit_tn is not None:
        parts.append(circuit_tn)
    tn = qtn.TensorNetwork(parts, check_collisions=False)
    probs = tn.contract(output_inds=[BATCH_IND], optimize=optimize, backend="jax").data
    return jnp.clip(jnp.real(probs), min=PROBABILITY_FLOOR)


def negative_log_likelihood(lpdo_tn: qtn.TensorNetwork, probs: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
    """``-mean(log p)`` over the batch, the KL divergence up to a data-dependent constant."""
    return -jnp.mean(jnp.log(probs))


def tp_penalty(lpdo_tn: qtn.TensorNetwork, probs: jnp.ndarray | None = None) -> jnp.ndarray:  # noqa: ARG001
    """Squared relative distance of the output-traced channel from the identity, ``‖Tr_out Λ − 𝟙‖_F² / 2**n``.

    Squaring keeps the gradient proportional to the violation, so large weights act as a smooth quadratic
    constraint. The likelihood alone rewards inflating the channel's trace, which this term counteracts; at
    weight ``w`` the equilibrium trace exceeds ``2**n`` by a factor ``1 + 1/(2w)``.
    """
    num_sites = len(lpdo_tn.tensors) // 2
    traced = lpdo_tn.reindex({output_ind(j): output_ind(j, dagger=True) for j in range(num_sites)})
    identity = qtn.TensorNetwork(
        [qtn.Tensor(jnp.eye(2), inds=(input_ind(j), input_ind(j, dagger=True))) for j in range(num_sites)]
    )
    distance = jnp.real(qtn.tensor_network_distance(traced, identity, backend="jax"))
    return distance**2 / 2**num_sites
