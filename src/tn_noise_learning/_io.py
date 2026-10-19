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
