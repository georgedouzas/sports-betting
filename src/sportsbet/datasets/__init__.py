"""It provides the tools to extract sports betting data."""

from __future__ import annotations

from ._base._dataloader import BaseDataLoader, load_dataloader
from ._base._schema import (
    BaseOddsSchema,
    BaseStatsSchema,
    optional_col,
    required_col,
)
from ._soccer._dataloader import SoccerDataLoader
from ._soccer._dummy import DummySoccerDataLoader

__all__: list[str] = [
    'BaseDataLoader',
    'BaseOddsSchema',
    'BaseStatsSchema',
    'DummySoccerDataLoader',
    'SoccerDataLoader',
    'load_dataloader',
    'optional_col',
    'required_col',
]
