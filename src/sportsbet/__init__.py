"""A collection of sports betting AI tools.

It provides classes to extract sports betting data and create predictive models. It contains two main
submodules:

- [`dataloaders`][sportsbet.dataloaders]: Turn what the sources carry into data to model.
- [`sources`][sportsbet.sources]: Where the data comes from, and the store that keeps it.
- [`evaluation`][sportsbet.evaluation]: Provides the classes to create and evaluate sports betting predictive models.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Param = dict[str, Any]
ParamGrid = dict[str, list[Any]] | list[dict[str, list[Any]]]
TrainData = tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]
FixturesData = tuple[pd.DataFrame, None, pd.DataFrame]
Data = NDArray[np.float64]
BoolData = NDArray[np.bool_]
Indices = NDArray[np.intp]
Schema = list[tuple[str, type[int] | type[float] | type[object] | type[np.datetime64]]]
OutputsMapping = dict[str, dict[str, Callable[..., pd.DataFrame]]]
