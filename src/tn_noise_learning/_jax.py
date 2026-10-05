"""Enable 64-bit JAX before any array exists; single precision underflows the trace-preservation penalty."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
