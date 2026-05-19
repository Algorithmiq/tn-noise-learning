"""Smoke test: the top-level public API is importable and `__all__` is reachable."""

from __future__ import annotations


def test_top_level_imports():
    import tn_noise_learning as tnnl

    for name in tnnl.__all__:
        assert hasattr(tnnl, name), f"{name} listed in __all__ but missing from module"
