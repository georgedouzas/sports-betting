"""A collection of sports betting AI tools.

Extract sports betting data and create predictive models with three submodules:

- [`sources`][sportsbet.sources]: Where the data comes from, and the store that keeps it.
- [`dataloaders`][sportsbet.dataloaders]: Turn what the sources carry into data to model.
- [`evaluation`][sportsbet.evaluation]: Create and evaluate sports betting predictive models.
"""

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

# The factory re-export sits after the type aliases because `_factory` imports them at load time.
from ._factory import (  # noqa: E402
    BuildError,
    build_bettor,
    build_dataloader,
    build_venue,
)

__all__ = [
    'BuildError',
    'build_bettor',
    'build_dataloader',
    'build_venue',
]
