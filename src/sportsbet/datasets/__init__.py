"""It provides the tools to extract sports betting data."""

from __future__ import annotations

from ._base._dataloader import BaseDataLoader, load_dataloader
from ._base._factory import from_dataframe, from_snapshots
from ._base._schema import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)
from ._soccer._dataloader import SoccerDataLoader
from ._soccer._dummy import DummySoccerDataLoader
from ._soccer._utils import market_outcomes

__all__: list[str] = [
    'BaseDataLoader',
    'BaseOddsSchema',
    'BaseStatsSchema',
    'DummySoccerDataLoader',
    'SoccerDataLoader',
    'from_dataframe',
    'from_snapshots',
    'load_dataloader',
    'market_outcomes',
    'optional_col',
    'required_col',
]
