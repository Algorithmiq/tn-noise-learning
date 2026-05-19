"""Smoke test: build a QPT MPO + sample a handful of shots."""

from __future__ import annotations

import qiskit as qk

from tn_noise_learning.qpt import build_qpt_mpo, simulate_experiments


def test_build_qpt_mpo_no_noise_smoke():
    qc = qk.QuantumCircuit(2)
    qc.cx(0, 1)
    qpt_mpo, states, povms = build_qpt_mpo(qc_unitary=qc)
    assert states.shape == (2, 6, 2, 2)
    assert povms.shape == (2, 6, 2, 2)
    # Two physical legs per qubit (state + povm hyper-legs).
    assert int(len(qpt_mpo.shape) / 2) == 2


def test_simulate_experiments_returns_correct_number_of_shots():
    qc = qk.QuantumCircuit(2)
    qc.cx(0, 1)
    qpt_mpo, _states, _povms = build_qpt_mpo(qc_unitary=qc)
    out = simulate_experiments(qpt_mpo, num_shots=3, joblib_workers=1)
    assert len(out) == 3
    for v in out.values():
        in_str, out_str = next(iter(v.items()))
        assert len(in_str) == 2
        assert len(out_str) == 2
