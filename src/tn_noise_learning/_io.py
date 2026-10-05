"""Typed accessors for zarr groups."""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr


def read_array(group: zarr.Group, name: str) -> np.ndarray:
    """Load the whole array ``name`` of ``group`` into memory."""
    array = group[name]
    if not isinstance(array, zarr.Array):
        raise TypeError(f"{name!r} is not an array in {group}.")
    return np.asarray(array[:])


def read_attr(group: zarr.Group, name: str) -> Any:
    """Attribute ``name`` of ``group``."""
    return group.attrs[name]
