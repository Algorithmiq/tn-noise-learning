"""Loss functions for LPDO training: KL divergence + trace-preservation penalty."""

from __future__ import annotations

import jax
import quimb.tensor as qtn
from jax import config
from jax import numpy as jnp
from quimb.tensor import TensorNetwork

# Enable double-precision JAX globally — single precision underflows badly on
# the trace-preservation penalty for χ > 2.
config.update("jax_enable_x64", True)


def kl_divergence(probs: jnp.ndarray) -> jnp.ndarray:
    r"""Negative-log-likelihood form of the KL divergence used by Torlai et al.

    For a batch of empirical transition probabilities ``probs``, returns
    ``-(1/N) Σ_i log p_i``.
    """
    return -1.0 * jnp.mean(jnp.log(probs))


def tp_penalty(self: TensorNetwork, join_outer_bonds: dict) -> jnp.ndarray:
    r"""Distance between the ancilla-traced LPDO and the identity — the TP constraint as a soft penalty.

    Returns ``‖Tr_aux[Λ(θ)] − 𝟙‖_F / 2^{n/2}``.
    """
    num_qubits = len(self.tensors) // 2

    id_mpo = qtn.MPO_identity(num_qubits)
    id_mpo.set_params(
        {0: jnp.array([jnp.eye(2)])}
        | {key: jnp.array([[jnp.eye(2)]]) for key in range(1, num_qubits - 1)}
        | {num_qubits - 1: jnp.array([jnp.eye(2)])}
    )

    return jnp.real(
        qtn.tensor_network_distance(self.reindex(join_outer_bonds), id_mpo, backend="jax") / (2 ** (num_qubits / 2))
    )


def loss_fn(params: dict, batch_state_params: dict, batch_povm_params: dict, skeletons):
    r"""Combined training loss ``L(θ) = KL(P_true | P_LPDO(θ)) + 1.2 · TP_penalty(θ)``.

    Returns ``(total_loss, (probs, nll_loss, tp_penalty_loss))`` so callers can
    log the individual components.
    """
    kraus_tn = qtn.unpack(params, skeletons[0])
    batch_state = qtn.unpack(batch_state_params, skeletons[1])
    batch_povm = qtn.unpack(batch_povm_params, skeletons[2])

    reindex_outer = {
        kraus_tn.outer_upper_ind_id.format(_): kraus_tn.outer_lower_ind_id.format(_) for _ in range(kraus_tn.L)
    }
    reindex_inner = {
        kraus_tn.inner_upper_ind_id.format(_): kraus_tn.inner_lower_ind_id.format(_) for _ in range(kraus_tn.L)
    }
    kraus_tn_dag = (kraus_tn.H).reindex(reindex_outer | reindex_inner)

    lpdo = qtn.TensorNetwork([kraus_tn, kraus_tn_dag])

    tn_transition_prob = qtn.TensorNetwork(
        [lpdo, batch_state, batch_povm],
        check_collisions=False,
    )

    probs = jnp.clip(
        jnp.real(tn_transition_prob.contract(output_inds=["exp"], backend="jax").data),
        min=1e-15,
    )

    nll_loss = kl_divergence(probs)
    tp_penalty_loss = tp_penalty(lpdo, reindex_outer)

    return nll_loss + 1.2 * tp_penalty_loss, (probs, nll_loss, tp_penalty_loss)


def compute_metrics(params, skeletons, test_data=None, true_lpdo=None, other_metrics=None) -> dict:
    """Evaluate downstream metrics: held-out loss, channel fidelity, LPDO trace.

    Args:
        params: kraus tensor parameters (output of `qtn.pack`).
        skeletons: tuple ``(kraus_skel, state_skel, povm_skel)``.
        test_data: list of ``(state_params, povm_params)`` batches for held-out evaluation.
        true_lpdo: ground-truth LPDO for channel-fidelity comparison.
        other_metrics: optional ``{name: callable}`` of custom metrics that
            take the reconstructed LPDO as their first argument.
    """
    import numpy as np

    from ..tn_core.lpdo_utils import build_lpdo, lpdo_fidelity, trace_lpdo

    lpdo = build_lpdo(tn_data=params.values())

    batch_loss = jax.jit(loss_fn, static_argnums=[3])
    if test_data is not None:
        mean_test_loss = np.mean(
            [
                batch_loss(params, batch_state_params, batch_povm_params, skeletons)[0]
                for batch_state_params, batch_povm_params in test_data
            ]
        )
    else:
        mean_test_loss = None

    metrics = {
        "frob_distance": lpdo_fidelity(lpdo, true_lpdo) if true_lpdo is not None else np.array([0]).sum(),
        "lpdo_trace": trace_lpdo(lpdo),
        "test_loss": mean_test_loss,
    }

    if other_metrics is not None:
        for key, func in other_metrics.items():
            if isinstance(func, list):
                func, args = func
            metrics = {key: func(lpdo, *args)} | metrics

    return metrics
