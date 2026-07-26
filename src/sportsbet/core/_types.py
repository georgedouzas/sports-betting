"""Define the types shared across the library."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Param: TypeAlias = dict[str, Any]
ParamGrid: TypeAlias = dict[str, list[Any]] | list[dict[str, list[Any]]]
TrainData: TypeAlias = tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]
FixturesData: TypeAlias = tuple[pd.DataFrame, None, pd.DataFrame]
Data: TypeAlias = NDArray[np.float64]
BoolData: TypeAlias = NDArray[np.bool_]
Indices: TypeAlias = NDArray[np.intp]
Schema: TypeAlias = list[tuple[str, type[int] | type[float] | type[object] | type[np.datetime64]]]
OutputsMapping: TypeAlias = dict[str, dict[str, Callable[..., pd.DataFrame]]]
