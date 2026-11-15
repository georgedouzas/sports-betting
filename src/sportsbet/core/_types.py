"""Define the types shared across the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type ParamGrid = dict[str, list[Any]] | list[dict[str, list[Any]]]
type TrainData = tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]
type FixturesData = tuple[pd.DataFrame, None, pd.DataFrame]
type Data = NDArray[np.float64]
type BoolData = NDArray[np.bool_]
type Indices = NDArray[np.intp]
