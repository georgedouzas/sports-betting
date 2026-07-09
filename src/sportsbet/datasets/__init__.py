"""It provides the tools to extract sports betting data."""

from __future__ import annotations

from ._base._dataloader import BaseDataLoader, load_dataloader
from ._base._schema import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)
from ._dummy import DummySoccerDataLoader
from ._soccer._dataloader import SoccerDataLoader
from ._soccer._schema import SoccerOddsSchema, SoccerStatsSchema

__all__: list[str] = [
    'BaseDataLoader',
    'BaseOddsSchema',
    'BaseStatsSchema',
    'DummySoccerDataLoader',
    'SoccerDataLoader',
    'SoccerOddsSchema',
    'SoccerStatsSchema',
    'load_dataloader',
    'optional_col',
    'required_col',
]
