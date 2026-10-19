from __future__ import annotations

import importlib.metadata

import jax

import tn_noise_learning


def test_version_matches_metadata():
    assert tn_noise_learning.__version__ == importlib.metadata.version("tn_noise_learning")


def test_public_names_resolve():
    for name in tn_noise_learning.__all__:
        assert getattr(tn_noise_learning, name) is not None


def test_x64_enabled_on_import():
    assert jax.config.jax_enable_x64
