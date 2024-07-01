"""A collection of sports betting AI tools.

It provides classes to extract sports betting data and create predictive models. It contains two main
submodules:

- [`datasets`][sportsbet.datasets]: Provides the classes to extract sports betting datasets.
- [`evaluation`][sportsbet.evaluation]: Provides the classes to create and evaluate sports betting predictive models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from nptyping import Bool, Float, Int, NDArray, Shape

Param = dict[str, Any]
ParamGrid = dict[str, list[Any]] | list[dict[str, list[Any]]]
TrainData = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]
FixturesData = tuple[pd.DataFrame, None, pd.DataFrame | None]
Data = NDArray[Shape['*, *'], Float]
BoolData = NDArray[Shape['*, *'], Bool]
Indices = NDArray[Shape['*, *'], Int]
Schema = list[tuple[str, type[int] | type[float] | type[object] | type[np.datetime64]]]
Outputs = list[tuple[str, Callable[..., pd.DataFrame]]]
